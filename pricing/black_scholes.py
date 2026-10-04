import numpy as np
import scipy.stats as st

def black_scholes_price(S,K,T,r, sigma, option_type):
    '''
    Calculate the Black-Scholes price for a European option.
    Recibe un precio de Strike del subyacente en el momento inicial (S), precio de Strike de la opción (K), el tiempo en años (T), la tasa de interés free risk en decimales (r), la volatilidad del subyacente en decimales (sigma) y el option type que puedes ser tanto una call como una put.
    Esta función devuelve el precio justo de una opción europea a través del modelo de Black-Scholes.
    En caso de lanzar un ValueError, se debe a que se ha insertado un tipo de opción inválido.
    '''
    d1 = (np.log(S/K) + (r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma *np.sqrt(T)
    descuento = np.exp(-r*T)
    option_type = option_type.lower()
    call = S*st.norm.cdf(d1) - K*descuento*st.norm.cdf(d2)
    put = K*descuento*st.norm.cdf(-d2) - S*st.norm.cdf(-d1)
    if option_type == "call":
        return call
    elif option_type == "put":
        return put
    else: 
        raise ValueError("Option type must be either 'Call' or 'Put'")
    