"""
BSD-3

Supplementary code for the IEEE Access Publication "Advancements In Spectral 
Power Distribution Modeling Of Light-Emitting Diodes" 
DOI: 10.1109/ACCESS.2022.3197280 
by Simon Benkner, 
Laboratory for Adaptive Lighting Systems and Visual Processing, 
Department of Electrcical Engineering, 
Technical University of Darmstadt, Germany.

"""
import numpy as np

def gaussian(self, x: float, amplitude: float, 
             center: float, sigma: float) -> float:
    """
    Gaussian shape pdf.
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Standard Deviviation.
    :type sigma: float
    :return: Function result.
    :rtype: float

    """
    return ((amplitude/(max(self.tiny, self.s2pi*sigma)))
            * np.exp(-(1.0*x-center)**2 / max(self.tiny, (2*sigma**2))))


def gaussian_split(self, x: float, amplitude: float, 
                   center: float, sigma: float, sigmar: float) -> float:
    """
    Split Gaussian shape pdf.
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Left side standard Deviviation.
    :type sigma: float
    :param sigmar: Right side standard Deviviation.
    :type sigmar: float
    :return: Function result.
    :rtype: float

    """
    amp = amplitude / (max(self.tiny, self.s2pi * (sigma + sigmar) ) )
    left = (x < center)*np.exp(-(1.0*x-center)**2 / max(self.tiny, (2*sigma**2)))
    right = (x >= center)*np.exp(-(1.0*x-center)**2 / max(self.tiny, (2*sigmar**2)))
    return amp * (left + right)


def lorentzian_sec_ord(self, x: float, amplitude: float, 
                       center: float, sigma: float) -> float:
    """
    Second order Lorentzian shape pdf.
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Standard Deviviation.
    :type sigma: float
    :return: Function result.
    :rtype: float

    """
    return amplitude / (1 + ( (x-center) / sigma )**2 )**2

    
def logistic_power_peak(self, x: float, amplitude: float, 
                        center: float, sigma: float, s: float) -> float:
    """
    Logistic Power Peak shape pdf.
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Standard Deviviation.
    :type sigma: float
    :param s: Shape parameter.
    :type s: float
    :return: Function result.
    :rtype: float

    """
    pt1 = amplitude / s
    pt2 = ( 1 + np.exp( (x-center+sigma*np.log(s))/sigma ))**(-(s+1)/s)
    pt3 = np.exp( (x-center+sigma*np.log(s))/sigma )
    pt4 = ( s+1 )**( (s+1) / s )
    return pt1 * pt2 * pt3 * pt4
    

def logistic_asymm_peak(self, x: float, amplitude: float, 
                        center: float, sigma: float, s: float) -> float:
    """
    Logistic Assymectrial Peak shape pdf.
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Standard Deviviation.
    :type sigma: float
    :param s: Shape parameter.
    :type s: float
    :return: Function result.
    :rtype: float

    """
    pt1 = amplitude 
    pt2 = ( 1 + np.exp( (x-center+sigma*np.log(s)) / s ) )**( -1*(s+1) )
    pt3 = ( s**(-s) ) * (s+1)**(s+1)
    pt4 = np.exp( (x-center+sigma*np.log(s)) / sigma )
    return pt1 * pt2 * pt3 * pt4


def pearson_vii(self, x: float, amplitude: float, 
                center: float, sigma: float, s: float) -> float:
    """
    Pearson Type VII like shape pdf. Note that this shape differs from the 
    common Type VII implementation due to its "skew" parameter s. 
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Standard Deviviation.
    :type sigma: float
    :param s: Skew parameter.
    :type s: float
    :return: Function result.
    :rtype: float

    """
    pt1 = amplitude
    pt2 = (1 + ( ( (x-center) / sigma )**2) * ( (2**(1/s)) - 1 ) )**s
    return pt1 / pt2


def pearson_vii_split(self, x: float, amplitude: float, center: float, 
                      sigma: float, sigmar: float, s: float, 
                      sr: float) -> float:
    """
    Split Pearson Type VII like shape pdf. Note that this shape differs from the 
    common Type VII implementation due to its "skew" parameter s. 
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Left side standard Deviviation.
    :type sigma: float
    :param sigmar: Right side standard Deviviation.
    :type sigmar: float
    :param s: Left side skew parameter.
    :type s: float
    :param sr: Right side skew parameter.
    :type sr: float
    :return: Function result.
    :rtype: float

    """
    pt1 = amplitude
    pt2 = (x < center) * ( ( (x-center) / sigma )**2) * ( (2**(1/s)) - 1 )
    pt3 = (x >= center) * ( ( (x-center) / sigmar )**2) * ( (2**(1/sr)) - 1 )
    pt4 = ( (x < center)*s + (x >= center)*sr )
    return pt1 / ( 1 + pt2 + pt3 )**(pt4)


def sigmoid_double_asymm(self, x: float, amplitude: float, center: float, 
                         sigma: float, s1: float, s2: float) -> float:
    """
    Sigmoid Double Assymetrical shape pdf.
    :param x: Dependent variable.
    :type x:  float
    :param amplitude: Amplitude.
    :type amplitude: float
    :param center: Mean / Peak wavelength.
    :type center: float
    :param sigma: Standard Deviviation.
    :type sigma: float
    :param s1: Left side shape parameter.
    :type s1: float
    :param s2: Right side shape parameter.
    :type s2: float
    :return: Function result.
    :rtype: float

    """
    pt1 = amplitude
    pt2 = ( 1 + np.exp( (x-center-0.5*sigma) / s1) ) 
    pt3 = ( 1 + np.exp( (x-center+0.5*sigma) / s2) )
    return ( pt1 / pt2 ) * ( 1 - ( 1 / (pt3) ) )