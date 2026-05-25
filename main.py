import numpy as np
from funtions.runtime import RUNTIME
from view import *
from view.animation  import solve_forward, conjugate_gradient

from preprocessing.preprocess  import load_data


def main():
    
    p = RUNTIME.get()
    save_data = p.save_dat
    animate = p.animate

    d = load_data()

    if p.run_type == "standard":
        Qout = p.Qout
        solve_forward(d, Qout, animate, save_data)

    elif p.run_type == "optimization":
        conjugate_gradient(d, animate, save_data, p)

    else:
        raise ValueError(f"run_type desconocido: {run_type}")


if __name__ == "__main__":
    main()
