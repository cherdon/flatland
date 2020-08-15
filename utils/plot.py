from __future__ import division
import os
import plotly
from plotly.graph_objects import Scatter
from plotly.graph_objs.scatter import Line
import torch


def plot_metric(x, y, title, path='plots'):
	'''
	Plots min, max and mean + standard deviation bars of a population over time, used to plot average reward.
	:param x:
	:param y:
	:param title: 
	:param path: 
	:return: 
	'''
	
	max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

	ys = torch.tensor(y, dtype=torch.float32)
	ys_min, ys_max, ys_mean, ys_std = ys.min(), ys.max(), ys.mean(), ys.std()
	
	ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

	trace_max = Scatter(x=x, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
	trace_upper = Scatter(x=x, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
	trace_mean = Scatter(x=x, y=ys.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
	trace_lower = Scatter(x=x, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
	trace_min = Scatter(x=x, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')
	
	plotly.offline.plot({
		'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
		'layout': dict(title=title, xaxis={'title': 'Trial'}, yaxis={'title': title})
	}, filename=os.path.join(path, "".join(title.split(" ")) + '.html'), auto_open=True)
