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
from spectral_fitting import (fit_everything, 
                              plot_test_spectra, 
                              evaluate_data, 
                              application_example_current, 
                              plot_cie_1976)


PATH_TEST_SPECTRA = r'test_spectra/different'
PATH_RESULTS = r'spectral_fitting/results/'
if __name__ == '__main__':
    plot_test_spectra()
    # fit_everything(PATH_TEST_SPECTRA, PATH_RESULTS)
    # res_mono = evaluate_data(r'results/mono')
    # res_pc = evaluate_data(r'results/pc')
    application_example_current()
    plot_cie_1976()