from collections import OrderedDict
import h5py
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import plot_mpl
from plotpal.file_reader import SingleTypeReader, match_basis
from scipy.interpolate import interp1d
import palettable

def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or):
    """
    Takes grid coordinates and data on grid and prepares it for 3D surface plotting in plotly

    Arguments:
    x_vals : NumPy array (1D) or float
        Gridspace x values of the data
    y_vals : NumPy array (1D) or float
        Gridspace y values of the data
    z_vals : NumPy array (1D) or float
        Gridspace z values of the data
    data_vals : NumPy array (2D)
        Gridspace values of the data

    Keyword Arguments:
    x_bounds : Tuple of floats of length 2
        If specified, the min and max x values to plot
    y_bounds : Tuple of floats of length 2
        If specified, the min and max y values to plot
    z_bounds : Tuple of floats of length 2
        If specified, the min and max z values to plot

    Returns a dictionary of keyword arguments for plotly's surface plot function

    """
    if type(x_vals) == np.ndarray and type(y_vals) == np.ndarray :
        yy, xx = np.meshgrid(y_vals, x_vals)
        zz = z_vals * np.ones_like(xx)
    elif type(x_vals) == np.ndarray and type(z_vals) == np.ndarray :
        zz, xx = np.meshgrid(z_vals, x_vals)
        yy = y_vals * np.ones_like(xx)
    elif type(y_vals) == np.ndarray and type(z_vals) == np.ndarray :
        zz, yy = np.meshgrid(z_vals, y_vals)
        xx = x_vals * np.ones_like(yy)

    if x_bounds is None:
        if type(y_vals) == np.ndarray and type(z_vals) == np.ndarray and bool_function == np.logical_or :
            x_bool = np.zeros_like(yy)
        else:
            x_bool = np.ones_like(yy)
    else:
        x_bool = (xx >= x_bounds[0])*(xx <= x_bounds[1])

    if y_bounds is None:
        if type(x_vals) == np.ndarray and type(z_vals) == np.ndarray and bool_function == np.logical_or :
            y_bool = np.zeros_like(xx)
        else:
            y_bool = np.ones_like(xx)
    else:
        y_bool = (yy >= y_bounds[0])*(yy <= y_bounds[1])

    if z_bounds is None:
        if type(x_vals) == np.ndarray and type(y_vals) == np.ndarray and bool_function == np.logical_or :
            z_bool = np.zeros_like(xx)
        else:
            z_bool = np.ones_like(xx)
    else:
        z_bool = (zz >= z_bounds[0])*(zz <= z_bounds[1])


    side_bool = bool_function.reduce((x_bool, y_bool, z_bool))


    side_info = OrderedDict()
    side_info['x'] = np.where(side_bool, xx, np.nan)
    side_info['y'] = np.where(side_bool, yy, np.nan)
    side_info['z'] = np.where(side_bool, zz, np.nan)
    side_info['surfacecolor'] = np.where(side_bool, data_vals, np.nan)

    return side_info

def generate_custom_colorbar():
    purples = palettable.colorbrewer.sequential.Purples_9
    purples = ['rgb({},{},{})'.format(*c) for c in purples.colors[1:]] #white to purple
    purples_nums = list(np.linspace(0.02, 0.45, len(purples)))
    cz_cbar = palettable.cmocean.sequential.Solar_4
    cz_cbar = ['rgb({},{},{})'.format(*c) for c in cz_cbar.colors]
    #cz_cbar.reverse()
    cz_cbar_nums = list(np.linspace(0.52, 1, len(cz_cbar)))

    new_cmap = []
    numbers = []
    new_cmap.append('rgb(255, 255, 255)')
    numbers.append(0)
    new_cmap.append('rgb(230, 230, 230)')
    numbers.append(0.02)
    for c, n in zip(purples, purples_nums):
        new_cmap.append(c)
        numbers.append(n)
    new_cmap.append('rgb(0, 0, 0)')
    numbers.append(0.5)
    new_cmap.append('rgb(30, 30, 30)')
    numbers.append(0.52)
    for c, n in zip(cz_cbar, cz_cbar_nums):
        new_cmap.append(c)
        numbers.append(n)

    my_scale = []
    for n, c in zip(numbers, new_cmap):
        my_scale.append((n, c))
    return my_scale, purples

color_scale, purples = generate_custom_colorbar()


in_dir = '../publication_invR10/triLayer_model_Pe3.2e3_Pr5e-1_tau5e-1_tauk03e-3_invR10_N2B10_Lx4_192x192x512-64_stitched/'
fig_name = 'plotly3D'

slice_reader = SingleTypeReader(in_dir, 'slices', fig_name, start_file=0, n_files=np.inf, distribution='even-write')
prof_reader = SingleTypeReader(in_dir, 'profiles', fig_name, start_file=0, n_files=np.inf, distribution='even-write')


stretch_factors = [0.988,] #for colorbar
cbar_edge_pad = 0.94
data_field = 'mu'
field_names = [data_field]

px_base = 1000
fontsize = int(16 * px_base/500)
x_pix, y_pix = px_base, px_base
cmaps = [color_scale]
colorbar_x = [0.9]
colorbar_dict=dict(lenmode='fraction', thicknessmode = 'fraction', len=0.25, thickness=0.02)
fig = go.Figure(layout={'width': x_pix, 'height': y_pix})
fig = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True},]], subplot_titles=field_names, horizontal_spacing=0.025)
scene_dict = {          'xaxis': {'showbackground':False, 'tickvals':[], 'title':''},
                        'yaxis': {'showbackground':False, 'tickvals':[], 'title':''},
                        'zaxis': {'showbackground':False, 'tickvals':[], 'title':''} }
fig.update_layout(scene = scene_dict,
                      margin={'l':0, 'r': 0, 'b':0, 't':0, 'pad':0},
                      font={'size' : fontsize, 'family' : 'Times New Roman'},
                      annotations={'font' : {'size' : fontsize, 'family' : 'Times New Roman'}})

field_bases = ['{}_x_side', '{}_x_mid', '{}_y_side', '{}_y_mid', '{}_z_2.5', '{}_z_0.5']
fields = [st.format(data_field) for st in field_bases]

if not slice_reader.idle:
    while slice_reader.writes_remain() and prof_reader.writes_remain():
        slice_dsets, plot_ind = slice_reader.get_dsets(fields)
        prof_dsets, prof_ind = prof_reader.get_dsets([data_field,])
        x = match_basis(slice_dsets[fields[-1]], 'x')
        y = match_basis(slice_dsets[fields[-1]], 'y')
        z = match_basis(slice_dsets[fields[0]], 'z')
        time_data = slice_dsets[fields[0]].dims[0]
        stretch_factor = stretch_factors[0]

        yz_side_data=slice_dsets['{}_x_side'.format(data_field)][plot_ind,:].squeeze()
        yz_mid_data= slice_dsets['{}_x_mid'.format(data_field)][plot_ind,:].squeeze()
        xz_side_data=slice_dsets['{}_y_side'.format(data_field)][plot_ind,:].squeeze()
        xz_mid_data= slice_dsets['{}_y_mid'.format(data_field)][plot_ind,:].squeeze()
        xy_side_data=slice_dsets['{}_z_2.5'.format(data_field)][plot_ind,:].squeeze()
        xy_mid_data= slice_dsets['{}_z_0.5'.format(data_field)][plot_ind,:].squeeze()

        z_profile = prof_dsets[data_field][prof_ind,:].squeeze()
        profile_func = interp1d(z, z_profile)

        x_max, x_mid, x_min = (x.max(), x[int(len(x)/2)], x.min())
        y_max, y_mid, y_min = (y.max(), y[int(len(y)/2)], y.min())
        z_max, z_mid, z_min = (2.5, 0.5, z.min())

        x_mid_off = x[x > x_mid][0]
        y_mid_off = y[y > y_mid][0]
        z_mid_off = z[z > z_mid][0]

        #Construct 1D outline lines
        lines = []
        constX_xvals = (x_min, x_max, x_mid, x_max, x_max, x_mid_off)
        constX_zvals = (z_max, z_max, z_max, z_min, z_mid, z_mid_off)
        constX_ybounds = ([y_min, y_max], [y_min, y_mid], [y_mid, y_max], [0, y_max], [y_mid, y_max], [y_mid, y_max])
        for x_val, y_bounds, z_val in zip(constX_xvals, constX_ybounds, constX_zvals):
            lines.append(OrderedDict())
            lines[-1]['y'] = np.linspace(*tuple(y_bounds), 2)
            lines[-1]['x'] = x_val*np.ones_like(lines[-1]['y'])
            lines[-1]['z'] = z_val*np.ones_like(lines[-1]['y'])

        constY_yvals = (y_min, y_max, y_mid, y_max, y_max, y_mid_off)
        constY_zvals = (z_max, z_max, z_max, z_min, z_mid, z_mid_off)
        constY_xbounds = ([x_min, x_max], [x_min, x_mid], [x_mid, x_max], [x_min, x_max], [x_mid, x_max], [x_mid, x_max])
        for x_bounds, y_val, z_val in zip(constY_xbounds, constY_yvals, constY_zvals):
            lines.append(OrderedDict())
            lines[-1]['x'] = np.linspace(*tuple(x_bounds), 2)
            lines[-1]['y'] = y_val*np.ones_like(lines[-1]['x'])
            lines[-1]['z'] = z_val*np.ones_like(lines[-1]['x'])

        constZ_xvals = (x_min, x_max, x_mid_off, x_mid, x_max)
        constZ_yvals = (y_max, y_min, y_mid_off, y_max, y_mid)
        constZ_zbounds = ([z_min, z_max], [z_min, z_max], [z_mid_off, z_max], [z_mid, z_max], [z_mid, z_max])
        for x_val, y_val, z_bounds in zip(constZ_xvals, constZ_yvals, constZ_zbounds):
            lines.append(OrderedDict())
            lines[-1]['z'] = np.linspace(*tuple(z_bounds), 2)
            lines[-1]['x'] = x_val*np.ones_like(lines[-1]['z'])
            lines[-1]['y'] = y_val*np.ones_like(lines[-1]['z'])


        #Construct top-ledoux lines
        top_ledoux_lines = []
        constX_xvals = (x_max, x_mid_off)
        constX_zvals = (2, 2)
        constX_ybounds = ([y_min, y_mid_off], [y_mid_off, y_max])
        line_width = (10, 5)
        for x_val, y_bounds, z_val, width in zip(constX_xvals, constX_ybounds, constX_zvals, line_width):
            top_ledoux_lines.append(OrderedDict())
            top_ledoux_lines[-1]['y'] = np.linspace(*tuple(y_bounds), 2)
            top_ledoux_lines[-1]['x'] = x_val*np.ones_like(lines[-1]['y'])
            top_ledoux_lines[-1]['z'] = z_val*np.ones_like(lines[-1]['y'])
            top_ledoux_lines[-1]['line'] = {'color':purples[-2], 'width' : width}

        constY_yvals = (y_max, y_mid_off)
        constY_zvals = (2, 2)
        constY_xbounds = ([x_min, x_mid_off], [x_mid_off, x_max])
        line_width = (10, 5)
        for x_bounds, y_val, z_val, width in zip(constY_xbounds, constY_yvals, constY_zvals, line_width):
            top_ledoux_lines.append(OrderedDict())
            top_ledoux_lines[-1]['x'] = np.linspace(*tuple(x_bounds), 2)
            top_ledoux_lines[-1]['y'] = y_val*np.ones_like(lines[-1]['x'])
            top_ledoux_lines[-1]['z'] = z_val*np.ones_like(lines[-1]['x'])
            top_ledoux_lines[-1]['line'] = {'color':purples[-2], 'width' : width}

        xy_side = construct_surface_dict(x, y, z_max, xy_side_data, x_bounds=(x_min, x_mid), y_bounds=(y_min, y_mid))
        xz_side = construct_surface_dict(x, y_max, z, xz_side_data, x_bounds=(x_min, x_mid), z_bounds=(z_min, z_mid))
        yz_side = construct_surface_dict(x_max, y, z, yz_side_data, y_bounds=(y_min, y_mid), z_bounds=(z_min, z_mid))

        yz_mid = construct_surface_dict(x_mid, y, z, yz_mid_data, y_bounds=(y_mid, y_max), z_bounds=(z_mid, z_max), bool_function=np.logical_and)
        xy_mid = construct_surface_dict(x, y, z_mid, xy_mid_data, x_bounds=(x_mid, x_max), y_bounds=(y_mid, y_max), bool_function=np.logical_and)
        xz_mid = construct_surface_dict(x, y_mid, z, xz_mid_data, x_bounds=(x_mid, x_max), z_bounds=(z_mid, z_max), bool_function=np.logical_and)

        for d in [xz_side, xz_mid, yz_side, yz_mid]:
            for key in ['x', 'y', 'surfacecolor', 'z']:
                d[key][d['z'] > z_max] = np.nan
        surface_dicts = [xz_side, yz_side, yz_mid, xz_mid, xy_mid, xy_side]

        tickvals = ticktext = None
        if data_field == 'mu':
            cmin = 0
            cmax = profile_func(0.01)
            global_max = 0
            #Figure out gloal scale factor of perturbations
            for d in surface_dicts:
                edit = d['surfacecolor']
                edit_points = edit[edit > stretch_factor*cmax]
                if len(edit_points) == 0:
                    d['surfacecolor'] += 0.01
                    continue
                edit_points -= stretch_factor*cmax
                if np.nanmax(edit_points) > global_max:
                    global_max = np.nanmax(edit_points)
            #Rescale perturbations
            new_global_max = 0
            for d in surface_dicts:
                edit = d['surfacecolor']
                edit_points = edit[edit > stretch_factor*cmax]
                if len(edit_points) == 0:
                    d['surfacecolor'] += 0.01
                    continue
                edit_points -= stretch_factor*cmax
                edit_points /= global_max
                edit_points *= (cmax*stretch_factor - cmin)
                edit_points += stretch_factor*cmax
                d['surfacecolor'][edit > stretch_factor*cmax] = edit_points
                if np.nanmax(d['surfacecolor']) > new_global_max:
                    new_global_max = np.nanmax(d['surfacecolor'])
            for d in surface_dicts:
                d['surfacecolor'] /= (new_global_max * cbar_edge_pad)
                #print(np.nanmax(d['surfacecolor']), np.nanmin(d['surfacecolor']))
            tickvals = [0, 0.5, cmax]
            ticktext = ['{:.3f}'.format(t) for t in [0, stretch_factor*cmax, cmax]]
            cmax = 1
            #print(np.nanmin(xz_side['surfacecolor']), np.nanmax(xz_side['surfacecolor']))


        colorbar_dict['tickvals'] = tickvals
        colorbar_dict['ticktext'] = ticktext
        colorbar_dict['outlinecolor'] = 'black'
        colorbar_dict['xanchor'] = 'center'
        colorbar_dict['x'] = colorbar_x[0]
        colorbar_dict['y'] = 0.5
        colorbar_dict['tickfont'] = {'family' : "Times New Roman"}
        colorbar_dict['outlinecolor'] = 'black'
        colorbar_dict['outlinewidth'] = 3
        for d in surface_dicts:
            d['cmin'] = cmin
            d['cmax'] = cmax
            d['colorbar'] = colorbar_dict
            d['colorscale'] = cmaps[0]
            d['showscale'] = False
            d['lighting'] = {'ambient' : 1}

        xy_side['showscale'] = True

        for surface_dict in surface_dicts:
            good_vals = surface_dict['surfacecolor'][surface_dict['z'] > 2.25]
#            if good_vals.shape[0] != 0:
#                print(np.nanmax(good_vals), np.nanmin(good_vals))
            fig.add_trace(go.Surface(**surface_dict), 1, 1)

        for line_dict in lines:
            fig.add_trace(go.Scatter3d(**line_dict, mode='lines', line={'color':'black'}, showlegend=False), 1, 1)

        for line_dict in top_ledoux_lines:
            fig.add_trace(go.Scatter3d(**line_dict, mode='lines', showlegend=False), 1, 1)

    #        break
    #    break

        #https://stackoverflow.com/questions/63386812/plotly-how-to-hide-axis-titles-in-a-plotly-express-figure-with-facets
        for anno in fig['layout']['annotations']:
            anno['text'] = ''

        annotation_xvals = [colorbar_x[0] - 0.02]
        for xv in annotation_xvals:
            fig.add_annotation(
                x=xv,
                y=0.65,
                text="$\huge{\mu}$",
                showarrow=False,
                font=dict(
                    family="Times New Roman",
                    size=int(1.5*fontsize),
                    color="black"
                    ),
                align="center"
                )

        title_xvals = [0.5]
        labels = ['sim time = {:.3e}'.format(time_data['sim_time'][plot_ind]),]
        for xv, label in zip(title_xvals, labels):
            fig.add_annotation(
                x=xv,
                y=0.85,
                text=label,
                showarrow=False,
                font=dict(
                    family="Times New Roman",
                    size=int(1.5*fontsize),
                    color="black"
                    ),
                align="center"
                )

        #normalized from 0 or -1 to 1; not actually in data x, y, z units.
        viewpoint = {'camera_eye' : {'x' : 2*1.1, 'y': 0.4*1.1, 'z' : 0.5*1.1}
                    }
        fig.update_layout(scene = viewpoint)
        pio.write_image(fig, '{}/{}_{:06d}.png'.format(slice_reader.out_dir, fig_name, time_data['write_number'][plot_ind]), width=x_pix, height=y_pix, format='png', engine='kaleido')
        fig.data = []
