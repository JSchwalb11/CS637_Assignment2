import math
import numpy as np
from data_loader import get_data
import plotly.express as px

def expected_model(L_perp, d_e, rho_s, a=1, b=1, c=1, C_0=1,):
    L_perp = a*L_perp**(1/9)
    d_e__rho_s = ((b*d_e)*(c*rho_s))**(4/5)

    return C_0*(L_perp)*d_e__rho_s

if __name__ == '__main__':
    timestamps, X, derived_x, y = get_data(split=False)
    #fig = px.violin(X)
    """fig = px.histogram(X, color="sex",
                       marginal="box",  # or violin, rug
                       hover_data=range(len(X[1])-1))"""
    fig = px.histogram(X)
    fig.show()
