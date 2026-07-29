def print_config():
    import config

    print("\n" + "=" * 80, flush=True)
    print("CONFIG SETTINGS", flush=True)
    print("=" * 80, flush=True)

    for name in sorted(dir(config)):
        if name.startswith("_"):
            continue

        value = getattr(config, name)

        # Skip functions/classes
        if callable(value):
            continue

        # Skip imported modules
        if type(value).__name__ == "module":
            continue

        print(f"{name}: {value}", flush=True)

    print("=" * 80 + "\n", flush=True)