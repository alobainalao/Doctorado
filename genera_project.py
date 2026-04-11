import os

estructura = {
    "tesis_proyect": {
        "README.md": "",
        "requirements.txt": "",
        "pyproject.toml": "",
        ".gitignore": "",
        "configs": {
            "config1.json": "{}",
            "config2.json": "{}",
            "config3.json": "{}",
        },
        "data": {
            "input": {},
            "output": {},
        },
        "scripts": {
            "run_script1.py": "",
            "run_script2.py": "",
            "run_script3.py": "",
        },
        "src": {
            "core": {
                "__init__.py": "",
                "config": {
                    "__init__.py": "",
                    "loader.py": "",
                },
                "io": {
                    "__init__.py": "",
                    "reader.py": "",
                    "writer.py": "",
                },
                "processing": {
                    "__init__.py": "",
                    "cleaning.py": "",
                    "transformation.py": "",
                    "analysis.py": "",
                },
                "models": {
                    "__init__.py": "",
                    "model.py": "",
                },
                "pipeline": {
                    "__init__.py": "",
                    "runner.py": "",
                },
                "utils": {
                    "__init__.py": "",
                    "logger.py": "",
                    "helpers.py": "",
                },
                "constants": {
                    "__init__.py": "",
                    "settings.py": "",
                },
            }
        },
        "tests": {
            "__init__.py": "",
            "test_io.py": "",
            "test_processing.py": "",
            "test_pipeline.py": "",
        }
    }
}


def crear_estructura(base_path, estructura):
    for nombre, contenido in estructura.items():
        ruta = os.path.join(base_path, nombre)
        
        if isinstance(contenido, dict):
            os.makedirs(ruta, exist_ok=True)
            crear_estructura(ruta, contenido)
        else:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)


if __name__ == "__main__":
    crear_estructura(".", estructura)
    print("✅ Estructura creada correctamente")

