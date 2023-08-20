
counter = 0
def _LOGGER(tag, info, **kwargs):
    global counter
    counter += 1
    print('===============================================================')
    print(f'Counter: {counter}')
    print(f'TAG: {tag}')
    print(f'INFO: {info}')
    for key, value in kwargs.items():
        print(f"kwarg: {key}: {value}")
    print('===============================================================')