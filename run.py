from funtions.runtime import RUNTIME
from config.parameters import Parameters
import os

def run(modo=None):

    # Fuente de configuración:
    #   - app (modo=None + env vars seteadas): usa el ambiente del OS
    #   - terminal (modo=None sin vars de sim, o modo="dev"/"json"/etc.): usa default.json
    _APP_KEYS = {"run_type", "model", "metodo", "spacing", "dt"}
    if modo is None and any(k in os.environ for k in _APP_KEYS):
        env = dict(os.environ)   # app web: inyecta config por env vars
    else:
        env = None               # terminal: usa config/default.json

    RUNTIME.params = Parameters(env=env)

    from main import main
    return main()


if __name__ == "__main__":
    import sys
    modo = sys.argv[1] if len(sys.argv) > 1 else None
    #modo = "dev"
    run(modo)
