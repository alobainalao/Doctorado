from funtions.runtime import RUNTIME
from config.parameters import Parameters
import os

def run(modo=None):

    # 🔥 si no te pasan env directamente, lo tomas del sistema
    if modo is None:
        env = dict(os.environ)
    else:
        env = None

    # 🔥 aquí decides fuente
    RUNTIME.params = Parameters(env=env)

    from main import main
    return main()


if __name__ == "__main__":
    import sys
    modo = sys.argv[1] if len(sys.argv) > 1 else None
    #modo = "dev"
    run(modo)
