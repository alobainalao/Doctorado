from funtions.runtime import RUNTIME
from config.parameters import Parameters

def run(env=None):

    print("int run")
    # 🔥 aquí decides fuente
    RUNTIME.params = Parameters(env=env)

    from main import main
    return main()


if __name__ == "__main__":
    run()
