from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from matplotlib.colors import LightSource
import mayavi
from mayavi import mlab
from mayavi.tools.pipeline import grid_plane
import seaborn as sns


import palettable as pal
wes = pal.wesanderson.Zissou_5.hex_colors
wes.reverse()
margot = pal.wesanderson.Margot2_4.hex_colors
redgrey = sns.blend_palette([margot[0], margot[1]], as_cmap=True)

# Code from https://stackoverflow.com/questions/49098466/plot-3d-convex-closed-regions-in-matplotlib
# Accessed 13 June 2022

def prepare_convex(ax, halfspaces, feasible_point, alpha):
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    verts = hs.intersections
    hull = ConvexHull(verts)
    faces = hull.simplices

    # ls = LightSource(azdeg=225.0, altdeg=45.0)
    # colourRGB = np.array((192/255.0, 192/255.0 ,192/255.0, 1.0))
    # normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])
    # rgbNew = np.array([colourRGB*shade for shade in ls.shade_normals(normalsarray, fraction=1.0)])

    for s in faces:
        sq = [
        [verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]],
        [verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]],
        [verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]]
        ]

        f = a3.art3d.Poly3DCollection([sq])
        f.set_color("white")
        f.set_edgecolor('grey')
#        f.set_facecolor(rgbNew)
        f.set_alpha(alpha)
        ax.add_collection3d(f)
    return ax

def plot_3D_constrain_space_convex(halfspaces, feasible_point=np.array([0.1, 0.1, 0.1]), axeslim=[0,1.1], alpha=0.3, points=[]):
    fig = plt.figure()
    ax = a3.Axes3D(fig, auto_add_to_figure=False)
    ax.dist=10
    ax.azim=30
    ax.elev=10
    ax.set_xlim(axeslim)
    ax.set_ylim(axeslim)
    ax.set_zlim(axeslim)
    ax.set_xlabel("reaction 1")
    ax.set_ylabel("reaction 2")
    ax.set_zlabel("reaction 3")
    ax = prepare_convex(ax, halfspaces, feasible_point, alpha)
    if len(points) > 0:
        ax.scatter3D(points[:,0], points[:,1], points[:,2], color="red")
    fig.add_axes(ax)
    plt.show()

def plot_3D_constrain_space_concave(subshapes, feasible_points, axeslim=[0,1.1], alpha=0.1, points=[], altpoints=[]):
    points = np.array(points)
    altpoints = np.array(altpoints)
    fig = plt.figure()
    ax = a3.Axes3D(fig, auto_add_to_figure=False)
    ax.dist=10
    ax.azim=30
    ax.elev=10
    ax.set_xlim(axeslim)
    ax.set_ylim(axeslim)
    ax.set_zlim(axeslim)
    ax.set_xlabel("reaction 1")
    ax.set_ylabel("reaction 2")
    ax.set_zlabel("reaction 3")
    for sub,fp in zip(subshapes, feasible_points):
        ax = prepare_convex(ax, sub, fp, alpha)
    #fig.add_axes(ax)
    if len(points) > 0:
        ax.scatter3D(points[:,0], points[:,1], points[:,2], color="red")
    if len(altpoints) > 0:
        ax.scatter3D(altpoints[:,0], altpoints[:,1], altpoints[:,2], color="black")
    fig.add_axes(ax)
    plt.show()

def plotly_plot_convex(halfspace, feasible_point, points=[], cemax=1):
    hs = HalfspaceIntersection(halfspace, feasible_point)
    verts = hs.intersections
    hull = ConvexHull(verts)
    faces = hull.simplices
    fig = go.Figure(data=[
        go.Mesh3d(
        x=verts[:,0],
        y=verts[:,1],
        z=verts[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        opacity=1,
        color=margot[0],
        flatshading=True
        )
    ])
    fig.update_traces(lighting=dict(specular=0.9))
    fig.update_layout(scene = dict(
        xaxis = {"range":(0,cemax+0.05), "tickfont":{"size":14}, "tickangle":0},
        yaxis = {"range":(0,cemax+0.05), "tickfont":{"size":14}, "tickangle":0},
        zaxis = {"range":(0,cemax+0.05), "tickfont":{"size":14}, "tickangle":0},
        xaxis_title='Reaction 1',
        yaxis_title='Reaction 2',
        zaxis_title='Reaction 3'),
        font_size=16,
        )
    fig.show()

def plotly_plot_concave(halfspaces, feasible_points, points=[], points_aft=[], cemax=1):
    def _makemesh(verts, faces):
        mesh = go.Mesh3d(
        x=verts[:,0],
        y=verts[:,1],
        z=verts[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        opacity=1,
        color=margot[0],
        flatshading=True
        )
        return mesh

    def _sphere(x, y, z, radius=0.03, resolution=20, color=margot[1]):
        """Return the coordinates for plotting a sphere centered at (x,y,z)"""
        u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
        X = radius * np.cos(u)*np.sin(v) + x
        Y = radius * np.sin(u)*np.sin(v) + y
        Z = radius * np.cos(v) + z
        sphere = go.Mesh3d({
            "x": X.flatten(),
            "y": Y.flatten(),
            "z": Z.flatten(),
            "alphahull": 0,
            "flatshading":True,
            "color": color
            })
        return sphere

    all_verts = []
    all_faces = []
    for halfsp, fp in zip(halfspaces, feasible_points):
        hs = HalfspaceIntersection(halfsp, fp)
        verts = hs.intersections
        hull = ConvexHull(verts)
        faces = hull.simplices
        all_verts.append(verts)
        all_faces.append(faces)
    plotdata = []
    if len(points)>0:
        plotdata.extend([_sphere(points[i,0],points[i,1], points[i,2], color="black") for i in range(points.shape[0])])
    if len(points_aft)>0:
        plotdata.extend([_sphere(points_aft[i,0],points_aft[i,1], points_aft[i,2]) for i in range(points_aft.shape[0])])
    meshes = [_makemesh(v, f) for v,f in zip(all_verts, all_faces)]
    plotdata.extend(meshes)
    fig = go.Figure(data=plotdata)
    fig.update_traces(lighting=dict(specular=0.9))
#    fig.add_trace(go.Scatter3d(
#        x=[0.6],
#        y=[0.1],
#        z=[0.6],
#        mode="text",
#        name="",
#        text=["<b>B</b>"],
#        textposition="middle center",
#        textfont=dict(size=32)
#        ))
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=0.08) )
    fig.update_layout(scene_camera = camera)
    fig.update_layout(scene = dict(
        xaxis = {"range":(0,cemax+0.05), "tickfont":{"size":14}, "tickangle":0, "tickmode": "linear", "tick0": 0, "dtick": 0.2, "color": "black"},
        yaxis = {"range":(0,cemax+0.05), "tickfont":{"size":14}, "tickangle":0, "tickmode": "linear", "tick0": 0, "dtick": 0.2, "color": "black"},
        zaxis = {"range":(0,cemax+0.05), "tickfont":{"size":14}, "tickangle":0, "tickmode": "linear", "tick0": 0, "dtick": 0.2, "color": "black"},
        xaxis_title='Catalytic efficiency reaction 1',
        yaxis_title='Catalytic efficiency reaction 2',
        zaxis_title='Catalytic efficiency reaction 3'),
        font=dict(family="sans-serif"),
        font_size=16,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        autosize=False,
        width=100,
        height=100
        )
    fig.write_image("../figures/figure_3B.png", scale=4.2, width=1000, height=1000)


def mayavi_plot_convex(halfspaces, feasible_point, points=[]):
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    verts = hs.intersections
    hull = ConvexHull(verts)
    faces = hull.simplices
    x, y, z = zip(*verts)
    mlab.triangular_mesh(x,y,z,faces)
    if points:
        mlab.points3d(points[:,0],points[:,1],points[:,2])
    mlab.axes()
    mlab.show()

def mayavi_plot_concave(halfspaces, feasible_points):
        # Create three simple grid plane modules.

    mlab.figure(size = (1024,768), bgcolor = (1,1,1))
    for halfsp, fp in zip(halfspaces, feasible_points):
        hs = HalfspaceIntersection(halfsp, fp)
        verts = hs.intersections
        hull = ConvexHull(verts)
        faces = hull.simplices
        x, y, z = zip(*verts)
        mlab.triangular_mesh(x,y,z,faces, color=colors.to_rgb(margot[0]))
    # To make some decent axes
    xx = yy = zz = np.arange(0,1.2,0.2)
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)
    lensoffset = 0
    mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.01)
    mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.01)
    mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.01)
    mlab.text3d(0.5,0,0, "reaction 1", color=(0,0,0), scale=0.1, orientation=(0,90,0))
    mlab.text3d(0,0.5,0, "reaction 2", color=(0,0,0), scale=0.1, orientation=(90,90,0))
    mlab.text3d(0,0,0.5, "reaction 3", color=(0,0,0), scale=0.1, orientation=(0,0,0))
    mlab.text3d(1,0,0, "1", color=(0,0,0), scale=0.1)
    mlab.text3d(0,1,0, "1", color=(0,0,0), scale=0.1)
    mlab.text3d(0,0,1, "1", color=(0,0,0), scale=0.1)
#    mlab.axes()
    # First normal to 'x' axis.
    # gp = GridPlane()
    # mlab.pipeline.grid_plane(gp)
    # # Second normal to 'y' axis.
    # gp = GridPlane()
    # mlab.add_module(gp)
    # gp.grid_plane.axis = 'y'
    # # Third normal to 'z' axis.
    # gp = GridPlane()
    # mlab.add_module(gp)
    # gp.grid_plane.axis = 'z'
    mlab.show()

# def plot_3D_mayavi(shapes):
#     fig = mlab.figure()
#
#     ax_ranges = [-2, 2, -2, 2, 0, 8]
#     ax_scale = [1.0, 1.0, 0.4]
#     ax_extent = ax_ranges * np.repeat(ax_scale, 2)
#
#     surf3 = mlab.surf(mx, my, mz1, colormap='Blues')
#     surf4 = mlab.surf(mx, my, mz2, colormap='Oranges')
#
#     surf3.actor.actor.scale = ax_scale
#     surf4.actor.actor.scale = ax_scale
#     mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])
#     mlab.outline(surf3, color=(.7, .7, .7), extent=ax_extent)
#     mlab.axes(surf3, color=(.7, .7, .7), extent=ax_extent,
#               ranges=ax_ranges,
#               xlabel='x', ylabel='y', zlabel='z')


#w = np.array([1., 1., 1.])
# ∑ᵢ hᵢ wᵢ qᵢ - ∑ᵢ gᵢ wᵢ <= 0
#  qᵢ - ubᵢ <= 0
# -qᵢ + lbᵢ <= 0

strongsynergism = np.array([
    [-1.5,  1.0,  0.0],
    [ 1.0, -1.5,  0.0],
    [-1.0,  2.0, -1.0],
    [ 2.0, -1.0, -1.0],
])

weakantagonism = [np.array([[ 1.0,  1.5, -1.0]]),
                    np.array([[ 1.5,  1.0, -1.0]])]

synergism = np.array([
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

synergism_fp  = np.array([0.1,0.1,0.1])

# Three-way tradeoff
tradeoff = [

    np.array([
    # arm along axis x1
    [-1.0,  0.0,  0.0,  0.0], # all variables above zero
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    [ 0.0,  1.0,  0.0, -0.2], # x2 less than 0.2
    [ 0.0,  0.0,  1.0, -0.2], # x3 less than 0.2
    [ 1.0,  4.0,  0.0, -1.0], # tradeoff between x1 and x2, peak at x1
    [ 1.0,  0.0,  4.0, -1.0], # tradeoff between x1 and x3, peak at x1
    ]),
    np.array([
    # arm along axis x2
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    [ 1.0,  0.0,  0.0, -0.2],
    [ 0.0,  0.0,  1.0, -0.2],
    [ 4.0,  1.0,  0.0, -1.0],
    [ 0.0,  1.0,  4.0, -1.0],
    ]),
    np.array([
    # arm along axis x3
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    [ 1.0,  0.0,  0.0, -0.2],
    [ 0.0,  1.0,  0.0, -0.2],
    [ 4.0,  0.0,  1.0, -1.0],
    [ 0.0,  4.0,  1.0, -1.0],
    ])
]

tradeoff_fp = [ np.array([0.1, 0.1, 0.1]), # x1
                np.array([0.1, 0.1, 0.1]), # x2
                np.array([0.1, 0.1, 0.1]), # x3
              ]

# two-way tradeoff
tradeoff_x1x2 = [
    np.array([
    # x1 and x3
    # foundation: all above zero
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    # tradeoff between x1 and x3
    [ 1.0,  0.0,  4.0, -1.0], # peak at x1
    # x1 and x2 synergy
    [-3.0,  1.0,  0.0,  0.0],
    [ 1.0, -3.0,  0.0,  0.0],
    [-1.0,  2.0,  0.0, -1.0],
    [ 2.0, -1.0,  0.0, -1.0],
    # x2 and x3 synergy
    [ 0.0, -3.0,  1.0,  0.0],
    [ 0.0,  1.0, -3.0,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
    ]),
    np.array([
    # foundation: all above zero
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    # tradeoff between x1 and x3
    [ 4.0,  0.0,  1.0, -1.0], # peak at x3
    # x1 and x2 synergy
    [-3.0,  1.0,  0.0,  0.0],
    [ 1.0, -3.0,  0.0,  0.0],
    [-1.0,  2.0,  0.0, -1.0],
    [ 2.0, -1.0,  0.0, -1.0],
    # x2 and x3 synergy
    [ 0.0, -3.0,  1.0,  0.0],
    [ 0.0,  1.0, -3.0,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
    ])
]

tradeoff_x1x2_mod = [
    np.array([
    # x1 and x3
    # foundation: all above zero
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    # tradeoff between x1 and x3
    [ 1.0,  0.0,  1.5, -1.0], # peak at x1
    # x1 and x2 synergy
    [-1.5,  1.0,  0.0,  0.0],
    [ 1.0, -1.5,  0.0,  0.0],
    [-1.0,  2.0,  0.0, -1.0],
    [ 2.0, -1.0,  0.0, -1.0],
    # x2 and x3 synergy
    [ 0.0, -1.5,  1.0,  0.0],
    [ 0.0,  1.0, -1.5,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
    ]),
    np.array([
    # foundation: all above zero
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    # tradeoff between x1 and x3
    [ 1.5,  0.0,  1.0, -1.0], # peak at x3
    # x1 and x2 synergy
    [-1.5,  1.0,  0.0,  0.0],
    [ 1.0, -1.5,  0.0,  0.0],
    [-1.0,  2.0,  0.0, -1.0],
    [ 2.0, -1.0,  0.0, -1.0],
    # x2 and x3 synergy
    [ 0.0, -1.5,  1.0,  0.0],
    [ 0.0,  1.0, -1.5,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
    ])
]

tradeoff_x1x2_fp = [ np.array([0.1, 0.1, 0.1]),
                     np.array([0.1, 0.1, 0.1]),
                   ]

tradeoff_x1x2_x1x3 = [
    np.array([
    # x1 and x3
    # foundation: all above zero
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    # tradeoff between x1 and x3
    [ 1.0,  0.0,  4.0, -1.0], # peak at x1
    # tradeoff between x1 and x2
    [ 1.0,  4.0,  0.0, -1.0], # peak at x1
    # x2 and x3 synergy
    [ 0.0, -3.0,  1.0,  0.0],
    [ 0.0,  1.0, -3.0,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
    ]),
    np.array([
    # foundation: all above zero
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    # tradeoff between x1 and x3
    [ 4.0,  0.0,  1.0, -1.0], # peak at x3
    # tradeoff between x1 and x2
    [ 4.0,  1.0,  0.0, -1.0], # peak at x2
    # x2 and x3 synergy
    [ 0.0, -3.0,  1.0,  0.0],
    [ 0.0,  1.0, -3.0,  0.0],
    [ 0.0, -1.0,  2.0, -1.0],
    [ 0.0,  2.0, -1.0, -1.0],
    ])
]
tradeoff_x1x2_x1x3_fp = [ np.array([0.1, 0.1, 0.1]),
                     np.array([0.1, 0.1, 0.1]),
                   ]
