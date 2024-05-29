import streamlit as st 
import torch
import numpy as np 
import plotly.graph_objects as go


#----------Global Variables-----------


torch.manual_seed(42)

lower_bound = -1.
upper_bound =  1.
sample_size = 11
feature_1 = torch.linspace(lower_bound,upper_bound,sample_size)
inputs_for_line = torch.linspace(lower_bound-2,upper_bound+2,sample_size)
ground_truth = feature_1 + 0.2 


#----------Scatter & Line Plot [aka Figure 1]-----------

def generate_plot(w):

  Datapoints = go.Scatter(
  x = feature_1,
  y = ground_truth ,
  mode = 'markers',
  name = 'Data points'
)
  
  line = go.Scatter(
      x = inputs_for_line,
      y = w * inputs_for_line + 0.2,
      mode = 'lines',
      name = 'model'
  )


  figure = go.Figure(data=[line, Datapoints])

  figure.update_layout(
    xaxis = dict(
      range = [-2,2],
      title = 'X',
      zeroline = True,
      zerolinewidth = 2,
      zerolinecolor = 'rgba(205, 200, 193, 0.7)'
    ),
    yaxis = dict(
      range = [-2,2],
      title = 'Y',
      zeroline = True,
      zerolinewidth = 2,
      zerolinecolor = 'rgba(205, 200, 193, 0.7)'
    ),
    height = 500,
    width = 2600
  )
  return figure

#----------Grid formation for the loss function----------
 

weight_combo = torch.linspace(-5,7,350)

L_F = []

for w in weight_combo:
  l = torch.mean((ground_truth-( w * feature_1 + 0.2))**2)
  L_F.append(l)


min_index = np.argmin(L_F)  
L_F = torch.tensor(L_F)

#----------Loss Landscape------------------------------------

def loss_landscape(w):

  grid = go.Scatter(
    x = weight_combo ,
    y = L_F,
    mode = 'lines+markers',
    opacity = 0.3,
    line = dict(color='pink'),
    marker = dict(size=10),
    name = 'Loss function landscape'
    
  )

  Global_minima = go.Scatter(
    x = (1,),
    y = (torch.min(L_F),),
    mode = 'markers',
    marker = dict(color='yellow',size=10,symbol='diamond'),
    name = 'Global minima'
  )

  ball = go.Scatter(
    x = (w,),
    y = (torch.mean((ground_truth-(w*feature_1 + 0.2))**2),),
    mode = 'markers',
    marker = dict(color='red',size = 7),
    name = 'loss'
  )

  layout = go.Layout(
     xaxis = dict(title='weight',range=[-4.5,6.5],zeroline=True,zerolinewidth=2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
     yaxis = dict(title= 'loss',range=[-2,10],zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
  )

  figure = go.Figure(data=[grid,Global_minima,ball],layout=layout)


        
  return figure


#------------------------------------------------------------------------------------------------------------------------------------------------------
# streamlit 

st.set_page_config(layout='wide')


st.title("Linear Regression: Single Feature, No Bias")
st.write('By : Hawar Dzaee')



with st.sidebar:
    st.subheader("Adjust the parameter to minimize the loss")
    w_val = st.slider("weight (w):", min_value=-4.0, max_value=4.0, step=0.1, value= -3.5)


container = st.container()

with container:
 
    st.write("")  # Add an empty line to create space

    # Create two columns with different widths
    col1, col2 = st.columns([3,3])

    # Plot figure_1 in the first column
    with col1:
        figure_1 = generate_plot(w_val)
        st.plotly_chart(figure_1, use_container_width=True, aspect_ratio=5.0)  # Change aspect ratio to 1.0
        st.latex(r'''\hat{y} = wX ''')

    # Plot figure_2 in the second column
    with col2:
        figure_2 = loss_landscape(w_val)
        st.plotly_chart(figure_2, use_container_width=True, aspect_ratio=5.0)
        st.latex(r"""\text{MSE(w)} = \frac{1}{n} \sum_{i=1}^n (\ y_i- (wX ) )^2""")


# -----------



