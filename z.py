from julia import Main
import numpy as np

a = np.zeros([1, 2, 3, 4])

Main.include("z2.jl")
Main.viewTensor(a)
