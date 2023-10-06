#!/usr/bin/env python3

import os
import copy
import scipy
import numpy as np
import pandas as pd
import itertools as itr
import scipy.optimize as opt
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
from scipy import stats
import pickle
import surfaceplot as sp

import palettable as pal
wes = pal.wesanderson.Zissou_5.hex_colors
wes.reverse()
margot = pal.wesanderson.Margot2_4.hex_colors
redgrey = sns.blend_palette([margot[0], margot[1]], as_cmap=True)

"""Modelling proteins with different degrees of reaction-based frustration"""

# Idea to subdivide nonconvex feasible spaces:
# https://math.stackexchange.com/questions/60416/optimization-over-union-of-convex-sets

def nonlinobj(x, ignored=[]):
    # Sum over the number of dimensions so that at a maximally efficient
    # enzyme has a fitness of one
    if len(ignored) > 0:
        return -sum([not ign for ign in ignored]) / sum([1/x[i] for i,ign in zip(range(len(x)), ignored) if not ign])
    else:
        return -len(x) / sum([1/x[i] for i in range(len(x))]) # Kacser & Burns

def productobj(x):
    return -np.product(x)

def solve_linear(obj, constraints):
    # constraints are given in the form -1*x1 + 2*x2 -1 =< 0
    # but linprog wants them in the form -1*x1 + 2*x2 =< 1
    # so change sign on the last term
    ineq_mat = constraints[:,:-1]
    ineq_vec = -constraints[:,-1]
    sol = opt.linprog(c=obj, A_ub=ineq_mat, b_ub=ineq_vec)
    if sol.status != 0:
        raise RuntimeError("Solver did not find solution!")
    else:
        return sol.fun, sol.x

def solve_nonlinear(obj, ineq, ignored=[]):
    varnum = ineq.shape[1]-1 # the number of variables/traits/reactions
    # This too needs to be solved twice because multiple equivalent solutions may exist
    bounds = tuple([ (0, np.inf) for i in range(varnum) ])
    # These constraints are taken to be non-negative, so -1 * the inequalities set up for the linear solver
    constr = tuple([{"type": "ineq", "fun": lambda x, c=c: sum([-c[i]*x[i] for i in range(varnum)]) - c[-1]} for c in ineq])
    sol = opt.minimize(obj, [0.1 for i in range(varnum)], method="SLSQP", bounds=bounds, constraints=constr, args=(ignored,))
    if sol.status != 0:
        raise RuntimeError("Solver did not find solution!")
    else:
        return sol.fun, sol.x

# helper functions
def _nonlinearmax(x, fit, ignored):
    # Keep the objective constant (that is, the fitness)
    return nonlinobj(x, ignored) - fit + 1e-10

def _nonlinearmin(x, fit, ignored):
    # Keep the objective constant (that is, the fitness)
    return -nonlinobj(x, ignored) + fit + 1e-10

def _maxinonlinobj(x, ignored):
    # Allow for maximization instead of minimization
    return -np.sum([x[i] for i in range(len(x)) if not ignored[i]])

def _mininonlinobj(x, ignored):
    # Allow for minimization instead of maximization
    return np.sum([x[i] for i in range(len(x)) if not ignored[i]])

def variability_analysis(fit, point, constraints, objective=[], problem="linear", ignored=[]):
    # Based on Mahadevan & Shilling (2003)
    if type(constraints) != np.ndarray: # a list -- space is concave. Solve seperately
        n = constraints[0][0].shape[0]-1 # Number of dimensions
        variability = np.full((n,2), np.nan)
        for constr in constraints:
            # It may happen that the point investigated is not feasible in the
            # subspace examined. Check this before continuing (are any constraints
            # violated?)
            inspace = []
            for row in constr:
                # Due to small errors in the solution, tiny
                # constraint violations can occur. Must check
                # if the points are within this error or not
                ins1 = 0 >= np.sum([row[i]*point[i] for i in range(len(point))]) + row[-1] + 1e-6
                ins2 = 0 >= np.sum([row[i]*point[i] for i in range(len(point))]) + row[-1] - 1e-6
                # the point must lie outside the error (upper and lower) to be considered a violation
                inspace.append(ins1 or ins2)
            if all(inspace):
                # If false, continue to next subspace
                minmax = variability_analysis(fit, point, constr, objective, problem, ignored)
                # Find the maximum range (the range may extend from one subspace
                # to another)
                for i in range(minmax.shape[0]):
                    # The left values are max, the right values are min of the range
                    # Replace the original nans with a value
                    upper = np.nanmax([variability[i,0], minmax[i, 0]])
                    lower = np.nanmin([variability[i,1], minmax[i, 1]])
                    variability[i] = np.array([upper, lower])
    else:
        n = constraints[0].shape[0]-1 # Number of dimensions
        variability = []
        for i in range(n):
            minmax = []
            if problem == "linear":
                # The inequalities that form the feasible space
                ineq_mat = constraints[:,:-1]
                ineq_vec = -constraints[:,-1]
                # The equality keeping fitness at its maximum
                eq_mat = np.array([objective])
                eq_vec = np.array([fit])
                obj = [0 for j in range(n)]
                # Find the maximum permitted value of dimension i
                obj[i] = -1
                options = {"maxiter": 50000, "tol": 1e-5}
                solmax = opt.linprog(c=obj, A_ub=ineq_mat, b_ub=ineq_vec, A_eq=eq_mat, b_eq=eq_vec, options=options)
                if solmax.status != 0:
                    raise RuntimeError("Solver did not find solution!")
                minmax.append(solmax.x[i])
                # Find the minimum permitted value of dimension i
                obj[i] = 1
                solmin = opt.linprog(c=obj, A_ub=ineq_mat, b_ub=ineq_vec, A_eq=eq_mat, b_eq=eq_vec, options=options)
                if solmin.status != 0:
                    raise RuntimeError("Solver did not find solution!")
                minmax.append(solmin.x[i])
            elif problem == "nonlinear":
                bounds = tuple([ (0, np.inf) for j in range(n) ])
                # constraints arising from the feasible space
                constrfs = [{"type": "ineq", "fun": lambda x, c=c: sum([-c[j]*x[j] for j in range(n)]) - c[-1]} for c in constraints]
                # Add equality constraint - solution to objective must remain at maximum
                constrobj = [{"type": "ineq", "fun": _nonlinearmax, "args": (fit, ignored,)},
                             {"type": "ineq", "fun": _nonlinearmin, "args": (fit, ignored,)}
                            ]
                constr = tuple(constrfs + constrobj)
                ign = [True]*n
                ign[i] = False
                solmax = opt.minimize(_maxinonlinobj, point, method="SLSQP", bounds=bounds, constraints=constr, args=(ign,), options={'ftol':1e-5, 'eps': 1e-10, "maxiter":100000})
                if solmax.status != 0:
                    raise RuntimeError("Solver did not find solution!")
                minmax.append(solmax.x[i])
                solmin = opt.minimize(_mininonlinobj, point, method="SLSQP", bounds=bounds, constraints=constr, args=(ign,), options={'ftol': 1e-5,'eps': 1e-10, "maxiter":100000})
                if solmin.status != 0:
                    raise RuntimeError("Solver did not find solution!")
                minmax.append(solmin.x[i])
            variability.append(minmax)
    variability = np.array(variability)
    if np.isnan(variability).any():
        raise RuntimeError("Nans found!")
    return variability

def return_optimal(fit, pos, abs_tol=1e-6):
    # find the maximum fitness, then all solutions that are at that fitness
    # minimization, so find the most negative value
    maxfit = min([sol for sol in fit])
    # Solver may have small errors (floating points). Use absolute tolerance
    optfit = [sol for sol in fit if abs(sol-maxfit) <= abs_tol]
    optpos = [pt for sol,pt in zip(fit,pos) if abs(sol-maxfit) <= abs_tol]
    # Solution may contain the same point multiple times. Return each point only once
    remove = []
    for (i,j) in itr.combinations(range(len(optpos)), 2):
        # Check if the points are equivalent or not
        eq = all([abs(p1-p2) <= abs_tol for p1,p2 in zip(optpos[i],optpos[j])])
        if eq:
            remove.append(j) # arbitrary choice as points equivalent
    optfit = [sol for k,sol in enumerate(optfit) if not k in remove]
    optpos = [pos for k,pos in enumerate(optpos) if not k in remove]
    return optfit, optpos

def solve_concave(objectives, constraints, problem="linear", ignored=[], abs_tol=1e-6):
    # Sanity check. The number of objectives and the number of constraint bundles
    # should be the same!
    if len(objectives) != len(constraints):
        raise RuntimeError("The number of objectives and the number of convex spaces should be the same!")
    fitness = []
    position = []
    for obj, constr in zip(objectives, constraints):
        if problem == "linear":
            fit, point = solve_linear(obj, constr)
        elif problem == "nonlinear":
            fit, point= solve_nonlinear(obj, constr, ignored)
        fitness.append(fit)
        position.append(list(point))
    optfit, optpos = return_optimal(fitness, position, abs_tol=abs_tol)
    return optfit, optpos

def solve_convex(objectives, constraints, problem="linear", ignored=[]):
    if problem == "linear":
        fit, point = solve_linear(objectives, constraints)
    elif problem == "nonlinear":
        fit, point = solve_nonlinear(objectives, constraints, ignored)
    return fit, [point]

def sum_conflicts(group, concave):
    sums = []
    new_concave = [pair for pair in concave if pair[0] in group and pair[1] in group]
    for p in group:
        count = sum([1 for c in new_concave if p in c])
        sums.append(count)
    return sums

def remove_conflicts(group, concave):
    group = sorted(group)
    conflict = [1]
    new_group = []
    while any([c > 0 for c in conflict]):
        # Pair all members of the old group
    #    comb = [pair for pair in itr.combinations(group,2)]
        # Find if any members are in conflict with one another
        # Calculate conflict scores
        conflict = sum_conflicts(group, concave)
        if any([c > 0 for c in conflict]):
            # Only resolve one conflict at a time
            index = np.argmax(conflict)
            # New group may contain conflicts
            expelled = group.pop(index)
            new_group.append(expelled)
    return group, new_group

def create_feasible_space(relationships):
    n = max([v for pair in relationships.keys() for v in pair])+1 # number of dimensions
    concave = []
    # foundation: all above zero (especially for plotting)
    constraints = np.zeros((n, n+1))
    constraints[:,0:-1] = np.diag([-1]*n)
    for pair,r in relationships.items():
        if r == "antagonism":
            concave.append(pair)
        elif r == "weakantagonism":
            constr = np.zeros((2,n+1))
            constr[:,pair[0]] = weakantagonism2d[:,0]
            constr[:,pair[1]] = weakantagonism2d[:,1]
            constr[:,-1] = weakantagonism2d[:,2]
            constraints = np.concatenate([constraints, constr])
        elif r == "unconstrained":
            constr = np.zeros((2,n+1))
            constr[:,pair[0]] = unconstrained2d[:,0]
            constr[:,pair[1]] = unconstrained2d[:,1]
            constr[:,-1] = unconstrained2d[:,2]
            constraints = np.concatenate([constraints, constr])
        elif r == "weaksynergism":
            constr = np.zeros((2,n+1))
            constr[:,pair[0]] = weaksynergism2d[:,0]
            constr[:,pair[1]] = weaksynergism2d[:,1]
            constr[:,-1] = weaksynergism2d[:,2]
            constraints = np.concatenate([constraints, constr])
        elif r == "synergism":
            constr = np.zeros((4,n+1))
            constr[:,pair[0]] = synergism2d[:,0]
            constr[:,pair[1]] = synergism2d[:,1]
            constr[:,-1] = synergism2d[:,2]
            constraints = np.concatenate([constraints, constr])
        else:
            raise InputError("Pairwise relationships must be of the kind 'synergism', \
            'weaksynergism', 'unconstrained', 'weakantagonism' or 'antagonism'.")
    peaks = {}
    if len(concave) > 0:
        # For the concave problems, divide equations up into compatible and
        # incompatible pairs
        fulltradeoff = len(concave) == scipy.special.comb(n, 2, exact=True)
        for pair in concave:
            c1 = np.zeros(n+1)
            c1[pair[0]] = antagonism2d[0][0][0]
            c1[pair[1]] = antagonism2d[0][0][1]
            c1[-1] = antagonism2d[0][0][2]
            c2 = np.zeros(n+1)
            c2[pair[0]] = antagonism2d[1][0][0]
            c2[pair[1]] = antagonism2d[1][0][1]
            c2[-1] = antagonism2d[1][0][2]
            # I do not need to care if the equations describe a peak or not.
            # The only thing that is important is that inequalities that are part
            # of the same concave pair do not end up in the same subspace
            for r,c in zip(pair, [c1,c2]):
                if r in peaks.keys():
                    peaks[r].append(c)
                else:
                    peaks[r] = [c]
            # c1 and c2 are not compatible (must be in seperate subspaces)
#        if fulltradeoff:
#            addconstr = []
#            for pair in concave:
#                ac1 = np.zeros(n+1)
#                ac2 = np.zeros(n+1)
#                ac1[pair[0]] = antagonism2d[0][0][0]
#                ac2[pair[1]] = antagonism2d[0][0][1]
        # Sometimes the arrangements are indeterminate: There are multiple equivalent
        # ways of dividing the feasible space into convex subspaces.
        # Place all peaks in a single group and remove conflicting peaks to another
        # group. Repeat with that group until all groups have no conflicts
        groupings = []
        group = sorted([v for v in peaks.keys()])
        new_group = [True]
        while len(new_group) > 0:
            # Continue until all conflicts are resolved and no new groups are made
            group, new_group = remove_conflicts(group, concave)
            groupings.append(group)
            group = new_group
        # Assemble the feasible space
        feasible_space = []
        for gr in groupings:
            group_constraints = np.concatenate([constraints]+[np.array([eq]) for peak in gr for eq in peaks[peak]])
            feasible_space.append(group_constraints)
        if fulltradeoff:
            # Quick and dirty solution: Assume trade off will always be symmetrical
            # If each dimension is its own peak (tradeoffs between all)
            # Add a set of constraints that each peak should not exceed the
            # intersection of the two equations forming the tradeoff along its "sides"
            # Get the intersection point (probably will always be same but keep it
            # flexible)
            lhs = np.concatenate([v[:,:-1] for v in antagonism2d])
            rhs = np.concatenate([v[:,-1] for v in antagonism2d])
            intersection = -scipy.linalg.solve(lhs, rhs)
            threshold = np.unique(np.round(intersection,6))
            if len(threshold) > 1:
                raise RuntimeError("Tradeoff is not symmetrical")
            # Which equation has its peak along the x-axis?
#            prep = [row * [1,0,1] for row in antagonism2d]
#            xmax = [-row[0][-1] / row[0][0] for row in prep]
#            eqindex = np.argmax(xmax)
            new_feasible_space = []
            for subspace in feasible_space:
                index = np.argmax(np.sum(subspace > 0, axis=0)) # index of peak
                # constrain all others!
                addconstr = []
                for i in range(n):
                    if not i == index:
                        row = np.zeros((n+1,))
                        row[i] = 1.0
                        row[-1] = -threshold[0]
                        addconstr.append(np.array([row]))
                new_subspace = np.concatenate([subspace]+addconstr)
                new_feasible_space.append(new_subspace)
            feasible_space = new_feasible_space
    else:
        feasible_space = np.array(constraints)
    return feasible_space

def generate_pairwise_relationships(elements, dimensions):
    relationships = []
    for comb in itr.combinations_with_replacement(elements, int(scipy.special.comb(dimensions,2))):
        relation = {}
        for pair, rel in zip(itr.combinations(range(dimensions), 2), comb):
            relation[pair] = rel
        relationships.append(relation)
    return relationships

def explore_relationships(dimensions, problem="linear"):
    relationships = generate_pairwise_relationships(elements, dimensions)
    fit_before = []
    fit_after = []
    pos_before = []
    pos_after = []
    var_before = []
    var_after = []
    relation = []
    for rel in tqdm(relationships):
        relation.append([r for r in rel.values()])
        # Generate the feasible space
        fs = create_feasible_space(rel)
        # Solve the case where selection acts on all functions in the same
        # enzyme
        if problem == "linear":
            objective = [-1] * dimensions
        elif problem == "nonlinear":
            objective = nonlinobj
        else:
            raise InputError("Problem not recognized")
        if type(fs) == np.ndarray: # Space is convex (no strong antagonism)
            fit, point = solve_convex(objective, fs, problem)
            # there will be only one solution
            var_before.append(variability_analysis(fit, point, fs, objective, problem))
        else:
            fit, point = solve_concave([objective for i in range(len(fs))], fs, problem)
            vars = []
            for i in range(len(fit)):
                vars.append(variability_analysis(fit[i], point[i], fs, objective, problem))
            var_before.append(vars)
        fit_before.append(fit)
        pos_before.append(point)
        # Solve the case where selection acts on a set of duplicates specialized
        # for one function
        # Perhaps also do so for intermediate sets of duplicates (how many duplications
        # are sufficient to maximize fitness?)
        ind_sol = []
        ind_pnt = []
        ind_var = []
        for i in range(dimensions):
            if problem == "linear":
                objective = [0] * dimensions
                objective[i] = -1
                ignored = []
            elif problem == "nonlinear":
                objective = nonlinobj
                ignored = [True] * dimensions
                ignored[i] = False
            if type(fs) == np.ndarray: # Space is convex (no strong antagonism)
                fit, point = solve_convex(objective, fs, problem, ignored)
                ind_sol.append(fit)
                ind_pnt.append(point[0]) # To make a neat array - no data is thrown away!
                # get the variability
                ind_var.append(variability_analysis(fit, point, fs, objective, problem, ignored))
            else:
                # because these solutions will always be unique (no conflicts anymore!)...
                fit, point = solve_concave([objective for j in range(len(fs))], fs, problem, ignored)
                # I can take these results out of their enclosing lists
                # (otherwise these lists may contain multiple alternative solutions)
                ind_sol.append(fit[0])
                ind_pnt.append(point[0])
                ind_var.append(variability_analysis(fit[0], point[0], fs, objective, problem, ignored))
        # Variability of post-specialization solutions
        fit_after.append(ind_sol)
        pos_after.append(ind_pnt)
        var_after.append(ind_var)
    results = {}
    results["fit_before"] = fit_before
    results["fit_after"]  = fit_after
    results["pos_before"] = pos_before
    results["pos_after"]  = pos_after
    results["var_before"] = var_before
    results["var_after"]  = var_after
    results["relation"]   = relation
    results["dims"]       = dimensions
    results["problem"]    = problem
    return results

def relationship_analysis(relationship, problem, dimensions):
    # Generate the feasible space
    fs = create_feasible_space(relationship)
    # Solve the case where selection acts on all functions in the same
    # enzyme
    if problem == "linear":
        objective = [-1] * dimensions
    elif problem == "nonlinear":
        objective = nonlinobj
    else:
        raise InputError("Problem not recognized")
    if type(fs) == np.ndarray: # Space is convex (no strong antagonism)
        fit_bef, point_bef = solve_convex(objective, fs, problem)
        # there will be only one solution
        var_bef = variability_analysis(fit_bef, point_bef, fs, objective, problem)
    else:
        fit_bef, point_bef = solve_concave([objective for i in range(len(fs))], fs, problem)
        var_bef = []
        for i in range(len(fit_bef)):
            var_bef.append(variability_analysis(fit_bef[i], point_bef[i], fs, objective, problem))
    # Solve the case where selection acts on a set of duplicates specialized
    # for one function
    # Perhaps also do so for intermediate sets of duplicates (how many duplications
    # are sufficient to maximize fitness?)
    fit_aft = []
    point_aft = []
    var_aft = []
    for i in range(dimensions):
        if problem == "linear":
            objective = [0] * dimensions
            objective[i] = -1
            ignored = []
        elif problem == "nonlinear":
            objective = nonlinobj
            ignored = [True] * dimensions
            ignored[i] = False
        if type(fs) == np.ndarray: # Space is convex (no strong antagonism)
            fit, point = solve_convex(objective, fs, problem, ignored)
            fit_aft.append(fit)
            point_aft.append(point[0]) # To make a neat array - no data is thrown away!
            # get the variability
            var_aft.append(variability_analysis(fit, point, fs, objective, problem, ignored))
        else:
            # because these solutions will always be unique (no conflicts anymore!)...
            fit, point = solve_concave([objective for j in range(len(fs))], fs, problem, ignored)
            # I can take these results out of their enclosing lists
            # (otherwise these lists may contain multiple alternative solutions)
            fit_aft.append(fit[0])
            point_aft.append(point[0])
            var_aft.append(variability_analysis(fit[0], point[0], fs, objective, problem, ignored))
    rel = [r for r in relationship.values()]
    return [rel, fit_bef, point_bef, var_bef, fit_aft, point_aft, var_aft]

def explore_relationships_multiprocessing(dimensions, problem="linear"):
    relationships = generate_pairwise_relationships(elements, dimensions)
    fit_before = []
    fit_after = []
    pos_before = []
    pos_after = []
    var_before = []
    var_after = []
    relation = []
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        tasks = [pool.apply_async(relationship_analysis, args=(rel,problem,dimensions,)) for rel in relationships]
        for t in tqdm(tasks):
            res = t.get()
            relation.append(res[0])
            fit_before.append(res[1])
            pos_before.append(res[2])
            var_before.append(res[3])
            fit_after.append(res[4])
            pos_after.append(res[5])
            var_after.append(res[6])
    results = {}
    results["fit_before"] = fit_before
    results["fit_after"]  = fit_after
    results["pos_before"] = pos_before
    results["pos_after"]  = pos_after
    results["var_before"] = var_before
    results["var_after"]  = var_after
    results["relation"]   = relation
    results["dims"]       = dimensions
    results["problem"]    = problem
    return results

def estimate_frustration(points, maxpos=1.):
    # There can be multiple solutions before duplication!
    # Divide difference between achieved and "optimal" activities by the number
    # of dimensions to be able to compare different dimensionalities
    points = np.array(points)
    if len(points.shape) == 1:
        # catalytic efficiencies before duplication
        frust = np.sum([maxpos - ce for ce in points]) / points.shape[0]
    else:
        # Only evaluate frustration for the reaction that has been selected for (optimized)
        frust = np.sum([maxpos - points[i,i] for i in range(points.shape[0])]) / points.shape[0]
    return frust

def get_frustration(results):
    frustbef = []
    frustaft = []
    for i in range(len(results["pos_before"])):
        # There may be multiple (equivalent fitness) solutions before duplication
        # Take the mean of these
        frustbef.append(np.sum([estimate_frustration(p) for p in results["pos_before"][i]]) / len(results["pos_before"][i]))
        frustaft.append(estimate_frustration(results["pos_after"][i]))
    return frustbef, frustaft

def promiscuity_index(point):
    # replace zeros with a very small number (otherwise get nan)
    point = np.array([v if v != 0. else 1e-10 for v in point ])
    n = len(point)
    I = - 1 / np.log(n) * np.sum(point/np.sum(point) * np.log(point/np.sum(point)))
    return I

def read_results(fname):
    with open(fname, "rb") as f:
        results = pickle.load(f)
    return results

def get_promiscuity(results, scenario="none"):
    prombef = []
    promaft = []
    for i in range(len(results["pos_before"])): # Number of combinations is the
        # same for both before and after. The number of solutions per combination
        # differs
        prombef.append([promiscuity_index(p) for p in results["pos_before"][i]])
        if scenario == "mutdecay":
            row = []
            for v in results["var_after"][i]:
                # Variability solutions always contain invariable position (due to
                # selection)
                # The minimum will return the minimum catalytic activity for
                # activies not under selection
                pos = [min(v[j]) for j in range(v.shape[0])]
                row.append(promiscuity_index(pos))
            promaft.append(row)
        elif scenario == "midpoint":
            row = []
            for v in results["var_after"][i]:
                # Midpoint of the variability range. If solution invariant returns
                # same as pos_after
                pos = [sum(v[j])/2 for j in range(v.shape[0])]
                row.append(promiscuity_index(pos))
            promaft.append(row)
        else:
            promaft.append([promiscuity_index(p) for p in results["pos_after"][i]])
    return prombef, promaft

def plot_catalytic_efficiencies(catpar):
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.hist(np.log10(catpar["cateff"]), bins=16, color=margot[0])
    plt.xlabel(r"Log$_{10}$ catalytic efficiency", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_empirical_promiscuity(promiscuity):
    fig = plt.figure(figsize=(6.3, 4.7))
    plt.hist(promiscuity["prom_indx"], bins=16, color=margot[0])
    plt.xlabel("Promiscuity index", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,1)
    plt.tight_layout()
    plt.show()

def plot_number_obs_empirical_promiscuity(promiscuity):
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.hist(promiscuity["nobs"], bins=promiscuity["nobs"].max()+1, color=margot[0])
    plt.xlabel("Number of substrates", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.yscale("log")
    plt.xscale("log")
#    plt.xticks(np.linspace(2,max(promiscuity["nobs"]),5), fontsize=14)
#    plt.xticks(np.linspace(2,26,9), fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_distribution_promiscuity(results):
    bef, aft = get_promiscuity(results)
    promaft = [v for va in aft for v in va]
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.hist(promaft, bins=16, color=margot[0], density=True)
    plt.xlabel("Promiscuity index", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_promiscuity_maxcateff(promiscuity):
    print(stats.pearsonr(promiscuity["prom_indx"], np.log10(promiscuity["max_ce"])))
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.scatter(promiscuity["prom_indx"], np.log10(promiscuity["max_ce"]), color=margot[0])
    plt.xlabel("Promiscuity index", fontsize=16)
    plt.ylabel(r"log$_{10}$ maximum"+ "\ncatalytic efficiency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def get_correlation_reaction_cateffs(catpar, exclude = ["ATP","NAD+", "NADPH", "NADH", "NADP+"]):
    # This uses all catalytic parameters with nobs >= 2. Promiscuity index
    # has not yet undergone cleaning. Double check!
    exclude = [s.lower() for s in exclude]
    rxn1 = []
    rxn2 = []
    prom = []
    # Entries without protein id
    enz1 = set([(row[0],row[1],row[2]) for row in catpar.loc[:,["species","ec","species_id", "protein_id"]].values if row[3]=="None"])
    # Entries with protein id
    enz2 = [prid for prid in catpar["protein_id"].unique() if prid!="None"]
    for (sp,ec,org) in enz1: # Each unique combination we deem a single enzyme
        ce = catpar.loc[(catpar["species"]==sp) & (catpar["ec"]==ec) & (catpar["species_id"]==org), "cateff"].values
        subs = catpar.loc[(catpar["species"]==sp) & (catpar["ec"]==ec) & (catpar["species_id"]==org), "substrate"].values
        remove = [sb in exclude for sb in subs]
        if any(remove):
            ce = [c for c,i in zip(ce,remove) if not i]
        if len(ce) > 2:
            r1, r2 = np.random.choice(ce, 2, replace=False)
            rxn1.append(r1)
            rxn2.append(r2)
            prom.append(promiscuity_index([r1, r2]))
        elif len(ce) == 2:
            rxn1.append(ce[0])
            rxn2.append(ce[1])
            prom.append(promiscuity_index(ce))
        else:
            pass
    for prid in enz2:
        sp = catpar.loc[catpar["protein_id"]==prid, "species"].unique()
        ec = " and ".join(catpar.loc[catpar["protein_id"]==prid, "ec"].unique())
        org = catpar.loc[catpar["protein_id"]==prid, "species_id"].unique()
        ce =  catpar.loc[catpar["protein_id"]==prid, "cateff"].values
        subs = catpar.loc[catpar["protein_id"]==prid, "substrate"].values
        if len(sp) > 1:
            # This should not happen (it does though!) Reject proteins that do this
            pass
        # Remove substrates excluded from analysis (cofactors)
        remove = [sb in exclude for sb in subs]
        if any(remove):
            ce = [c for c,i in zip(ce,remove) if not i]
        if len(ce) > 2:
            r1, r2 = np.random.choice(ce, 2, replace=False)
            rxn1.append(r1)
            rxn2.append(r2)
            prom.append(promiscuity_index([r1, r2]))
        elif len(ce) == 2:
            rxn1.append(ce[0])
            rxn2.append(ce[1])
            prom.append(promiscuity_index(ce))
        else:
            pass
    samples = pd.DataFrame([[r1, r2, p] for r1, r2, p in zip(rxn1, rxn2, prom)], columns=["cateff1", "cateff2", "prom"])
    return samples

def plot_correlation_reaction_cateffs(samples):
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.scatter(np.log10(samples["cateff1"]), np.log10(samples["cateff2"]),
        c=samples["prom"], cmap=redgrey, alpha=0.7, edgecolor="none")
    plt.xlabel(r"log$_{10}$ catalytic efficiency 1", fontsize=16)
    plt.ylabel(r"log$_{10}$ catalytic efficiency 2", fontsize=16)
    cbar = plt.colorbar(label="Promiscuity index")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label="Promiscuity index", size=16)
    cbar.set_alpha(1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def get_correlation_stats(samples):
    cateff = samples.loc[:, ["cateff1", "cateff2"]].values
    cemean = np.mean(np.log10(cateff))
    cevar = np.var(np.log10(cateff))
    corr = np.corrcoef(np.log10(cateff[:,0]),np.log10(cateff[:,1]))
    return cateff, cemean, cevar, corr

def correlated_sampling(samples, num=2):
    cateff, cemean, cevar, corr = get_correlation_stats(samples)
    if num > 2:
        # correlation matrix
        corrmat = np.full((num,num), corr[0,1])
        np.fill_diagonal(corrmat, 1)
        # The same standard devs for all
        stds = np.std(np.log10(cateff[:,0])) * np.std(np.log10(cateff[:,1]))
        cecov = corrmat * stds
    else:
        cecov = np.cov(np.log10(cateff[:,0]), np.log10(cateff[:,1]))
    # Get the samples
    corrsamples = 10**np.random.multivariate_normal([cemean for i in range(num)], cecov, 5000)
    unisamples = 10**np.random.normal(cemean, cevar, (5000,num))
    return corrsamples, unisamples

def correlated_sampling_all(samples, maxdim=8, directory="../results/random_sampling/"):
    for i in tqdm(range(2,maxdim+1)):
        corrs, uncorrs = correlated_sampling(samples, i)
        np.savetxt(directory+"correlated_{}.txt".format(i), corrs)
        np.savetxt(directory+"uncorrelated_{}.txt".format(i), uncorrs)

def read_correlated_sampling(directory="../results/random_sampling/", corr=True):
    prom = []
    emp = pd.read_csv("../data/promiscuity_index.csv")
    for i in range(emp.shape[0]):
        prom.append(["BRENDA", emp.loc[i, "prom_indx"]])
    for i in range(2,9):
        if corr:
            fname = "correlated_{}.txt".format(i)
        else:
            fname = "uncorrelated_{}.txt".format(i)
        samples = np.loadtxt(directory+fname)
        for j in range(samples.shape[0]):
            pr = promiscuity_index(samples[j])
            prom.append([i, pr])
    return pd.DataFrame(prom, columns=["Data", "Promiscuity index"])

def plot_correlated_sampling(prom):
    fig = plt.figure(figsize=(6.3, 4.7))
    ax = sns.violinplot(x="Data", y="Promiscuity index", data=prom, scale="width",
        palette=[margot[1]]+[margot[0] for i in range(2,9)])
    ax.set_ylabel("Promiscuity index", fontsize=16)
    ax.set_xlabel("Number of reactions", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0,1)
    plt.tight_layout()
    plt.show()

def compare_distributions(samples, promiscuity):
    mod_samples = np.array([promiscuity_index(row) for row in samples])
    emp_samples = promiscuity["prom_indx"].values
    print(stats.ks_2samp(data1=mod_samples, data2=emp_samples))

def plot_sel_favoured_promiscuity(results, fitch):
    bef, aft = get_promiscuity(results)
    prom = []
    for i in range(len(bef)):
        fitchmax = fitch.loc[i, "fitchmax"]
        # if the maximum fitness change is above zero, duplication & selection is
        # favoured, else use promiscuity from before duplication
        if fitchmax > 0:
            prom.extend([v for v in aft[i]])
        else:
            prom.extend([v for v in bef[i]])
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.hist(prom, bins=16, color=margot[0], density=True)
    plt.xlabel("Promiscuity index", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_dimensions_promiscuity(results):
    dimaft = [sum([v>1e-3 for v in p]) for i in range(len(results["pos_after"])) for p in results["pos_after"][i]]
    bef, aft = get_promiscuity(results, scenario="mutdecay")
    promaft = [v for va in aft for v in va]
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.scatter(dimaft, promaft, color=margot[0])
    plt.xlabel("Number of reactions", fontsize=16)
    plt.ylabel("Promiscuity index", fontsize=16)
    plt.xticks(range(1, results["dims"]+1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def get_correlation_syndist_reaction_cateffs(results):
    distances = {"synergism": 0, "weaksynergism": 0.25, "unconstrained": 0.5,
    "weakantagonism": 0.75, "antagonism": 1}
    xpos = [sum(distances[r] for r in rel) for rel in results["relation"]]
    syndist = []
    for i in range(len(results["pos_before"])):
        for j in range(len(results["pos_before"][i])):
            syndist.append(xpos[i])
#    meanbefpos = [np.mean(v) for v in results["pos_before"]]
    befpos = [np.mean(v) for va in results["pos_before"] for v in va]
    print(stats.pearsonr(syndist, befpos))
    print("N = {}".format(len(befpos)))

def get_correlation_promiscuity_maxce_model(results):
#    maxceaft = [np.mean([max(p) for p in results["pos_after"][i] ]) for i in range(len(results["pos_after"])) ]
#    maxcebef = [np.mean([max(p) for p in results["pos_before"][i] ]) for i in range(len(results["pos_before"])) ]
    maxceaft = [max(p) for i in range(len(results["pos_after"])) for p in results["pos_after"][i] ]
    maxcebef = [max(p) for i in range(len(results["pos_before"])) for p in results["pos_before"][i] ]
    bef,aft = get_promiscuity(results, scenario="mutdecay")
    promaft = [v for va in aft for v in va]
    prombef = [v for va in bef for v in va]
    print("Before duplication")
    print(stats.pearsonr(prombef, maxcebef))
    print("n={}".format(len(prombef)))
    print("After duplication")
    print(stats.pearsonr(promaft, maxceaft))
    print("n={}".format(len(promaft)))

def get_percentage_promiscuous(directory="../results/", threshold=0.8):
    reslist = [f for f in os.listdir(directory) if f.startswith("results_")]
    spec = []
    dim = []
    for resf in sorted(reslist):
        dim.append(int(resf.split(".")[0][-1]))
        results = read_results(directory+resf)
        bef, aft = get_promiscuity(results)
        promaft = np.array([v for va in aft for v in va])
        spec.append(np.sum(promaft >= threshold)/promaft.shape[0]*100)
    np.savetxt(directory+"dimensions_specific.txt", dim)
    np.savetxt(directory+"percentage_specific.txt", spec)

def plot_percentage_promiscuous(threshold=0.8):
    dim = np.loadtxt("../results/dimensions_specific.txt")
    spec = np.loadtxt("../results/percentage_specific.txt")
    # Empirical data
    prom = pd.read_csv("../data/promiscuity_index.csv")
    emp = np.sum(prom["prom_indx"] >= threshold)/prom.shape[0] * 100
    x = ["BRENDA", ""] + [str(int(v)) for v in dim]
    y = [emp, 0] + [v for v in spec]
    color = [margot[1], "black"] + [margot[0] for v in spec]
    fig = plt.figure(figsize=(4.8, 3.6))
    ax = fig.add_subplot(111)
    ax.bar(x, y, color=color)
    ax.vlines(1, 0, 80, "black", linestyles="dashed")
    ax.set_xlabel("Number of reactions", fontsize=16)
    ax.set_ylabel("Percentage of enzymes\nwith high promiscuity", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].set_visible(False)
    ax.set_ylim(0,75)
    plt.tight_layout()
    plt.show()

def get_fitness_change_duplication(results):
    # unpack results dictionary
    n = results["dims"]
    problem = results["problem"]
    distances = {"synergism": 0, "weaksynergism": 0.25, "unconstrained": 0.5,
                "weakantagonism": 0.75, "antagonism": 1}
    xpos = [sum([distances[r] for r in rel]) for rel in results["relation"]]
    if results["problem"] == "linear":
        # fitnesses after duplication must be recalculated because the fitnesses
        # given by the optimization are calculated only with respect to the reaction
        # the duplicate specializes in
        new_fit_aft = [np.sum(pa)/n for pa in results["pos_after"]] # this value is now positive
        new_fit_bef = [-np.min(fb) for fb in results["fit_before"]] # To make sure there is but a single value
        # Comparison: Duplication is only beneficial if the fitness of all duplicates
        # taken together exceeds that of the original protein (corrected for the
        # fact that there are now n copies of the gene)
        fitch = [fa - fb for fa,fb in zip(new_fit_aft, new_fit_bef)]
        # For complete antagonism, the above calculation appears to misrepresent
        # the situation...
        # Now the same, but taking into account that the solution for
        # individual reactions may have several equilvalent answers
        # In contrast, optimizing for all reactions at once is likely to never
        # produce a degenerate answer
        # The only place where fitness can change
        # is with the fitnesses after duplication, because these have been optimized
        # for one function at a time, and thus either lies on a point (no var) or
        # along an edge parallel to an ignored axis (along which there can be var).
        new_fit_aft_max = [np.sum(np.array(pa)[:,:,0])/n for pa in results["var_after"]]
        new_fit_aft_min = [np.sum(np.array(pa)[:,:,1])/n for pa in results["var_after"]]
        fitchmax = [fa - fb for fa,fb in zip(new_fit_aft_max, new_fit_bef)]
        fitchmin = [fa - fb for fa,fb in zip(new_fit_aft_min, new_fit_bef)]
    elif results["problem"] == "nonlinear":
        # fitnesses after duplication must be recalculated because the fitnesses
        # given by the optimization are calculated only with respect to the reaction
        # the duplicate specializes in
        # sum the catalytic efficiencies for each reaction
        sumcateff = [np.sum(np.vstack(pa), axis=0) for pa in results["pos_after"]]
        # use the fitness function to calculate organismal fitness
        new_fit_aft = [-nonlinobj(x/n) for x in sumcateff]
        new_fit_bef = [-np.min(fb) for fb in results["fit_before"]]
        fitch = [fa - fb for fa,fb in zip(new_fit_aft, new_fit_bef)]
        # The only place where fitness can change
        # is with the fitnesses after duplication, because these have been optimized
        # for one function at a time, and thus either lies on a point (no var) or
        # along an edge parallel to an ignored axis (along which there can be var).
        sumcateffmax = [np.sum(np.vstack([sub[:,0] for sub in pa]), axis=0) for pa in results["var_after"]]
        sumcateffmin = [np.sum(np.vstack([sub[:,1] for sub in pa]), axis=0) for pa in results["var_after"]]
        new_fit_aft_max = [-nonlinobj(x/n) for x in sumcateffmax]
        new_fit_aft_min = [-nonlinobj(x/n) for x in sumcateffmin]
        fitchmax = [fa - fb for fa,fb in zip(new_fit_aft_max, new_fit_bef)]
        fitchmin = [fa - fb for fa,fb in zip(new_fit_aft_min, new_fit_bef)]
    # how dissimilar are the pairwise relationships?
    reldis = [sum([abs(distances[r1]-distances[r2]) for r1,r2 in itr.combinations(rel,2)]) for rel in results["relation"]]
    # Degree of frustation
    frustbef, frustaft = get_frustration(results)
    change = {"combination "+str(i): [row[i] for row in results["relation"]] for i in range(len(results["relation"][0]))}
    change["fitbef"] = new_fit_bef
    change["fitaft"] = new_fit_aft
    change["fitaftmax"] = new_fit_aft_max
    change["fitaftmin"] = new_fit_aft_min
    change["nsolbef"] = [len(row) for row in results["pos_before"]]
    change["nsolaft"] = [len(row) for row in results["pos_after"]]
    change["syndist"] = xpos
    change["reldist"] = reldis
    change["fitch"] = fitch
    change["fitchmin"] = fitchmin
    change["fitchmax"] = fitchmax
    change["frustbef"] = frustbef
    change["frustaft"] = frustaft
    return pd.DataFrame(change)

def rand_jitter(arr):
    #https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plot_fitness_change_duplication(fitch):
    # Only some points need a range
    fitch["range"] =  abs(fitch["fitchmin"] - fitch["fitchmax"]) >= 1e-3
    bar = fitch.loc[fitch["range"]].reset_index()
    nobar = fitch.loc[fitch["range"] == False].reset_index()
    # For those observations with a range, use midpoints
    bar.loc[:,"halfrange"] = (bar.loc[:,"fitchmax"] - bar.loc[:,"fitchmin"])/2
    bar.loc[:,"midpoints"] = bar.loc[:,"halfrange"] + bar.loc[:,"fitchmin"]
    xnobar = rand_jitter(nobar["syndist"])
    xbar = rand_jitter(bar["syndist"])
    fig = plt.figure(figsize=(6.3, 4.7))
    plt.hlines(0., 0, max(fitch["syndist"]) + 0.1, linestyles="dashed", colors=margot[1], linewidth=1.5)
    plt.scatter(xnobar/fitch["syndist"].max(), nobar["fitch"], color=margot[0])
    plt.errorbar(xbar/fitch["syndist"].max(), bar["midpoints"], yerr=bar["halfrange"],
        color=margot[0], linewidth=0, elinewidth=0.5, dash_capstyle="butt", capsize=5,
        marker="o")
    plt.xlabel("Average antagonism $\hat{A}$", fontsize=16)
    plt.ylabel("Change in fitness", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.06, 1.06)
    plt.tight_layout()
    plt.show()

def get_catalytic_gain_specialization(results):
    distances = {"synergism": 0, "weaksynergism": 0.25, "unconstrained": 0.5,
                "weakantagonism": 0.75, "antagonism": 1}
    xpos = [sum([distances[r] for r in rel]) for rel in results["relation"]]
    n = results["dims"]
    gain = []
    for i in range(len(results["pos_before"])):
        change3D = []
        nsolbef = len(results["pos_before"][i])
        syndist = xpos[i]
        rel = results["relation"][i]
        for j in range(n):
            # Can have multiple solutions before duplication...
            general = [v[j] for v in results["pos_before"][i]]
            change1D = [results["pos_after"][i][j][j]-g for g in general]#np.mean([results["pos_after"][i][j][j]-g for g in general])
            #change3D.extend(change1D)
            gain.append([rel[0], rel[1], rel[2], nsolbef, syndist, np.mean(change1D)])
    catgain = pd.DataFrame(gain, columns=["combination 0", "combination 1", "combination 2", "nsolbef", "syndist", "mean_cat_change"])
    return catgain

def plot_catalytic_gain_specialization(catgain):
    print(stats.pearsonr(catgain["syndist"], catgain["main_cat_change"]))
    np.random.seed(0)
    xpos = rand_jitter(catgain["syndist"])
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.scatter(xpos/catgain["syndist"].max(), catgain["mean_cat_change"], color=margot[0])
    plt.xlabel(r"Average antagonism $\hat{A}$", fontsize=16)
    plt.ylabel("Change in catalysis", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
#    plt.xlim(-0.06, 1.06)
    plt.tight_layout()
    plt.show()

def plot_frustration_after_specialization(fitch):
    np.random.seed(0)
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.plot(rand_jitter(fitch["reldist"]/fitch["reldist"].max()), fitch["frustaft"], 'o', color=margot[0])
    plt.xlabel("Dissimilarity of\n"+r"pairwise relationships $D$", fontsize=16)
    plt.ylabel("Frustration after\nspecialization", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def count_classes_duplication_five(nmax=8, directory="../results/"):
    classes = []
    for n in range(3, nmax+1):
        fitch = pd.read_csv(directory+"fitness_change_dimensions{}.csv".format(n))
        # No frustration
        nofrust = sum(abs(fitch["frustbef"]) <= 1e-3)
        # Frustration resolvable and selection favours duplication (max: it *can* or *cannot* increase fitness)
        frustres1 = sum( (abs(fitch["frustbef"]) >= 1e-3) & (abs(fitch["frustaft"]) <= 1e-3) & (fitch["fitchmax"] > 0) )
        # Frustration resolvable but not selected
        frustres0 = sum( (abs(fitch["frustbef"]) >= 1e-3) & (abs(fitch["frustaft"]) <= 1e-3) & (fitch["fitchmax"] <= 0) )
        # Frustration not fully resolvable
#        frustnores = sum( fitch["frustaft"] >= 1e-3 )
        # Frustration not fully resolvable but selected
        frustnores1 = sum( (fitch["frustaft"] >= 1e-3) & (fitch["fitchmax"] > 0) )
        # Frustration not fully resolvable and not selected
        frustnores2 = sum( (fitch["frustaft"] >= 1e-3) & (fitch["fitchmax"] <= 0) )
        classes.append([n, nofrust, frustres1, frustres0, frustnores1, frustnores2])
    return classes

def plot_classes_five(classes):
    classes = np.array(classes)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.bar(classes[:,0], classes[:,1]/np.sum(classes[:,1:], axis=1)*100,
        label="no frustration", color=wes[0], edgecolor="black")
    ax.bar(classes[:,0], classes[:,2]/np.sum(classes[:,1:], axis=1)*100,
        bottom=classes[:,1]/np.sum(classes[:,1:], axis=1)*100,
        label="resolvable, selected", color=wes[3], edgecolor="black")
    ax.bar(classes[:,0], classes[:,3]/np.sum(classes[:,1:], axis=1)*100,
        bottom=np.sum(classes[:,1:3], axis=1)/np.sum(classes[:,1:], axis=1)*100,
        label="resolvable, not selected", color=wes[4], edgecolor="black")
    ax.bar(classes[:,0], classes[:,4]/np.sum(classes[:,1:], axis=1)*100,
        bottom=np.sum(classes[:,1:4], axis=1)/np.sum(classes[:,1:], axis=1)*100,
        label="not resolvable, selected", color=wes[1], edgecolor="black")
    ax.bar(classes[:,0], classes[:,5]/np.sum(classes[:,1:], axis=1)*100,
        bottom=np.sum(classes[:,1:5], axis=1)/np.sum(classes[:,1:], axis=1)*100,
        label="not resolvable, not selected", color=wes[2], edgecolor="black")
    ax.set_xticks(classes[:,0])
    ax.set_ylabel("Percentage of enzymes", fontsize=16)
    ax.set_xlabel("Number of reactions", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=12)
    plt.tight_layout()
    plt.show()

def count_classes_duplication(nmax=8, directory="../results/"):
    classes = []
    for n in range(3, nmax+1):
        fitch = pd.read_csv(directory+"fitness_change_dimensions{}.csv".format(n))
        # No frustration
        nofrust = sum(abs(fitch["frustbef"]) <= 1e-3)
        # Frustration resolvable
        frustres = sum( (abs(fitch["frustbef"]) >= 1e-3) & (abs(fitch["frustaft"]) <= 1e-3) )
        # Frustration not fully resolvable
        frustnores = sum( (fitch["frustaft"] >= 1e-3) & (fitch["fitchmax"] > 0) )
        classes.append([n, nofrust, frustres, frustnores])
    return classes

def plot_classes(classes):
    classes = np.array(classes)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.bar(classes[:,0], classes[:,1]/np.sum(classes[:,1:], axis=1)*100,
        label="no frustration", color=wes[0], edgecolor="black")
    ax.bar(classes[:,0], classes[:,2]/np.sum(classes[:,1:], axis=1)*100,
        bottom=classes[:,1]/np.sum(classes[:,1:], axis=1)*100,
        label="frustration, resolvable", color=wes[3], edgecolor="black")
    ax.bar(classes[:,0], classes[:,3]/np.sum(classes[:,1:], axis=1)*100,
        bottom=np.sum(classes[:,1:3], axis=1)/np.sum(classes[:,1:], axis=1)*100,
        label="frustration, not resolvable", color=wes[2], edgecolor="black")
    ax.set_xticks(classes[:,0])
    ax.set_ylabel("Percentage of enzymes", fontsize=16)
    ax.set_xlabel("Number of reactions", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=12)
    plt.tight_layout()
    plt.show()

def get_promiscuity_comparison(after=True, scenario="mutdecay"):
    prom = []
    emp = pd.read_csv("../data/promiscuity_index.csv")
    for i in range(emp.shape[0]):
        prom.append(["BRENDA", emp.loc[i, "prom_indx"]])
    for i in range(3,9):
        results = read_results("../results/results_dimensions{}.pkl".format(i))
        prombef, promaft = get_promiscuity(results, scenario)
        if after:
            for va in promaft:
            #    prom.append([i, np.mean(va)])
                for v in va:
                    prom.append([i, v])
        else:
            for va in prombef:
                for v in va:
                    prom.append([i, v])
    return pd.DataFrame(prom, columns=["Data", "Promiscuity index"])

def plot_promiscuity_comparison(prom):
    fig = plt.figure(figsize=(6.4, 4.7))
    ax = sns.violinplot(x="Data", y="Promiscuity index", data=prom, scale="width",
        palette=[margot[1]]+[margot[0] for i in range(3,9)], bw=0.3)
    ax.set_ylabel("Promiscuity index", fontsize=16)
    ax.set_xlabel("Number of reactions", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0,1)
    plt.tight_layout()
    plt.show()

def uniform_sampling():
    constr = {"strong antagonism": antagonism2d, "weak antagonism": weakantagonism2d,
        "unconstrained": unconstrained2d, "weak synergism": weaksynergism2d,
        "strong synergism": synergism2d}
    rand_enz = {}
    for c, cts in constr.items():
        samples = []
        while len(samples) < 5000:
            # Randomly generate candidate enzyme
            enz = np.random.uniform(0, 1, (2,))
            # check if proposed enzyme conforms to constraints
            match = []
            if c != "strong antagonism":
                for con in cts:
                    match.append(con[0]*enz[0] + con[1]*enz[1] + con[2] <= 0)
                if all(match):
                    samples.append(enz)
            else:
                # Slightly different rules for strong antagonism: Not concave!
                for con in cts:
                    match.append(con[0][0]*enz[0] + con[0][1]*enz[1] + con[0][2] <= 0)
                if any(match):
                    samples.append(enz)
        rand_enz[c] = samples
    # change to pandas dataframe [relationship, cateff 1, cateff 2, promiscuity]
    raen = []
    for c, enz in rand_enz.items():
        for e in enz:
            raen.append([c, e[0], e[1], promiscuity_index(e)])
    return pd.DataFrame(raen, columns=["relationship", "cat. eff. 1", "cat. eff. 2", "promiscuity"])

def plot_uniform_sampling(enz):
    fig = plt.figure(figsize=(23.5, 4.7))
    letters = ["A", "B", "C", "D", "E"]
    for i, rel in enumerate(enz["relationship"].unique()):
        subset = enz.loc[enz["relationship"]==rel]
        ax = plt.subplot2grid((1,5), (0,i), colspan=1)
        ax.hist(subset["promiscuity"], bins=16, color=margot[0])
        ax.set_xlabel("Promiscuity index", fontsize=16)
        ax.set_ylabel("Frequency", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.annotate(letters[i], xy=(0.01, 1.), xycoords="axes fraction",
                        xytext=(5,-5), textcoords="offset points",
                        ha="left", va="top", weight="bold", fontsize=25)
    plt.tight_layout()
    fig.savefig("../figures/constrained_uniform_sampling.pdf")

def plot_figure_1():
    # Read in data
    catpar = pd.read_csv("../data/merged_catpar_kegg.csv")
    promiscuity = pd.read_csv("../data/promiscuity_index.csv")
    samples = pd.read_csv("../data/random_cateff_samples.csv")
    # create figure
    fig = plt.figure(figsize=(12.6,9.5))
    # Panel A
    axA = plt.subplot2grid((2,2), (0,0), colspan=1)
    axA.hist(np.log10(catpar["cateff"]), bins=16, color=margot[0])
    axA.set_xlabel(r"Log$_{10}$ catalytic efficiency", fontsize=16)
    axA.set_ylabel("Frequency", fontsize=16)
    axA.tick_params(axis="both", labelsize=14)
    axA.annotate("A", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel B
    axB = plt.subplot2grid((2,2), (0,1), colspan=1)
    axB.hist(promiscuity["prom_indx"], bins=16, color=margot[0])
    axB.set_xlabel("Promiscuity index", fontsize=16)
    axB.set_ylabel("Frequency", fontsize=16)
    axB.tick_params(axis="both", labelsize=14)
    axB.annotate("B", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel C
    axC = plt.subplot2grid((2,2), (1,0), colspan=1)
    axC.hist(promiscuity["nobs"], bins=promiscuity["nobs"].max()+1, color=margot[0])
    axC.set_xlabel("Number of substrates", fontsize=16)
    axC.set_ylabel("Frequency", fontsize=16)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.tick_params(axis="both", labelsize=14)
    axC.annotate("C", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel D
    axD = plt.subplot2grid((2,2), (1,1), colspan=1)
    sc = axD.scatter(np.log10(samples["cateff1"]), np.log10(samples["cateff2"]),
        c=samples["prom"], cmap=redgrey, alpha=0.7, edgecolor="none")
    axD.set_xlabel(r"log$_{10}$ catalytic efficiency 1", fontsize=16)
    axD.set_ylabel(r"log$_{10}$ catalytic efficiency 2", fontsize=16)
    axD.tick_params(axis="both", labelsize=14)
    cbar = fig.colorbar(sc, label="Promiscuity index")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label="Promiscuity index", size=16)
    cbar.set_alpha(1)
    axD.annotate("D", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    plt.tight_layout()
    fig.savefig("../figures/figure_1.png", dpi = 300)
    plt.close()

def plot_figure_3B():
    results = read_results("../results/results_dimensions3.pkl")
    rel = generate_pairwise_relationships(elements, 3)
    fp = create_feasible_space(rel[4])
    pos = np.array(results["pos_before"][4])
    posaft = np.array(results["pos_after"][4])
    sp.plotly_plot_concave(fp, sp.tradeoff_fp, points=pos, points_aft=posaft, cemax=0.65)

def plot_figure_3AC():
    results = read_results("../results/results_dimensions3.pkl")
    fitch = pd.read_csv("../results/fitness_change_dimensions3.csv")
    catgain = get_catalytic_gain_specialization(results)
    # Prepare for plotting
    np.random.seed(0)
    xpos = rand_jitter(catgain["syndist"])
    fig = plt.figure(figsize=(4.8, 7.2))
    # Panel A
    axA = plt.subplot2grid((2,1), (0,0), colspan=1)
    axA.scatter(xpos/catgain["syndist"].max(), catgain["mean_cat_change"], color=margot[0])
    axA.set_xlabel(r"Average antagonism $\hat{A}$", fontsize=16)
    axA.set_ylabel("Change in catalysis", fontsize=16)
    axA.tick_params(axis="both", labelsize=14)
    axA.annotate("A", xy=(0.,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel C
    axC = plt.subplot2grid((2,1), (1,0), colspan=1)
    axC.plot(rand_jitter(fitch["reldist"]/fitch["reldist"].max()), fitch["frustaft"], 'o', color=margot[0])
    axC.set_xlabel("Dissimilarity of\n"+r"pairwise relationships $D$", fontsize=16)
    axC.set_ylabel("Frustration after\nspecialization", fontsize=16)
    axC.tick_params(axis="both", labelsize=14)
    axC.annotate("C", xy=(0.,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    plt.tight_layout()
    fig.savefig("../figures/figure_3AC.png", dpi = 300)
    plt.close()

def plot_figure_4():
    classes = count_classes_duplication()
    classes = np.array(classes)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.bar(classes[:,0], classes[:,1]/np.sum(classes[:,1:], axis=1)*100,
        label="no frustration", color=wes[0], edgecolor="black")
    ax.bar(classes[:,0], classes[:,2]/np.sum(classes[:,1:], axis=1)*100,
        bottom=classes[:,1]/np.sum(classes[:,1:], axis=1)*100,
        label="frustration, resolvable", color=wes[3], edgecolor="black")
    ax.bar(classes[:,0], classes[:,3]/np.sum(classes[:,1:], axis=1)*100,
        bottom=np.sum(classes[:,1:3], axis=1)/np.sum(classes[:,1:], axis=1)*100,
        label="frustration, not resolvable", color=wes[2], edgecolor="black")
    ax.set_xticks(classes[:,0])
    ax.set_ylabel("Percentage of enzymes", fontsize=16)
    ax.set_xlabel("Number of reactions", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=12)
    plt.tight_layout()
    fig.savefig("../figures/figure_4.png")

elements = ["synergism", "weaksynergism", "unconstrained", "weakantagonism", "antagonism"]

# These must be ordered, smaller index to the left!
relationships = {(0,1): "synergism",
                 (0,2): "synergism",
                 (1,2): "synergism",
                }

relation2 = {(0,1): "antagonism",
             (0,2): "antagonism",
             (1,2): "synergism",
            }

relation3 = {(0,1): "synergism",
             (0,2): "synergism",
             (1,2): "antagonism",
            }
relation4 = {(0,1): "antagonism",
             (0,2): "antagonism",
             (1,2): "antagonism",
            }

relation5 = {(0,1): "weakantagonism",
             (0,2): "weaksynergism",
             (1,2): "unconstrained",
            }

# Problem set: 2D synergism
obj = np.array([-1,-1])
# Constraints in the form -1*x1 + 2*x2 -1 =< 0
synergism2d = np.array([
            [-3,  1,  0],
            [-1,  2, -1],
            [ 1, -3,  0],
            [ 2, -1, -1],
            ])

# Problem set: 2D weakly synergistic
weaksynergism2d = np.array([
            [-1,  5, -4],
            [ 5, -1, -4],
            ])

# Problem set: 2D Unconstrained
obj = [-1,-1]
unconstrained2d = np.array([
            [ 0,  1, -1],
            [ 1,  0, -1],
            ])

# Problem set: 2D weakly antagonistic
weakantagonism2d = np.array([
            [ 1,  4, -4],
            [ 4,  1, -4],
            ])

# Problem set: 2D trade-off
obj = [-1,-1]
antagonism2d = [ np.array([ [ 1,  4, -1] ]),
               np.array([ [ 4,  1, -1] ])
             ]


synergism3d = np.array([
    # x1 and x2
    [-3.0,  1.0,  0.0,  0.0],
    [ 1.0, -3.0,  0.0,  0.0],
    [-1.0,  2.0,  0.0, -1.0],
    [ 2.0, -1.0,  0.0, -1.0],
    # x1 and x3
    [-3.0,  0.0,  1.0,  0.0],
    [ 1.0,  0.0, -3.0,  0.0],
    [-1.0,  0.0,  2.0, -1.0],
    [ 2.0,  0.0, -1.0, -1.0],
    # x2 and x3
    [ 0.0, -3.0,  1.0,  0.0],
    [ 0.0,  1.0, -3.0,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
])
