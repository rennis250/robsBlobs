import numpy as np

AliceWeight_mw = np.array([0.44, 0.46, 0.1])
AliceWeight_sw = np.array([0.41, 0.44, 0.15])

RGB_weights_aliceMonitor = np.array([43.8726816818746, 100.422336391086, 8.66744700006506])
RGB_weights_aliceMonitor = RGB_weights_aliceMonitor/RGB_weights_aliceMonitor.sum()

RGB_weights_radiance_aliceMonitor = np.array([0.334350089660000, 0.211504326128800, 0.143650257850000])
RGB_weights_radiance_aliceMonitor = RGB_weights_radiance_aliceMonitor/RGB_weights_radiance_aliceMonitor.sum()