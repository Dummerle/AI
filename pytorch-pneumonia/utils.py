import colorama

colorama.init()


def log(txt: str):
    print(f"[{colorama.Fore.GREEN}LOG{colorama.Fore.RESET}] " + txt)


def error(txt: str):
    print(f"[{colorama.Fore.RED}ERROR{colorama.Fore.RESET}] " + txt)
