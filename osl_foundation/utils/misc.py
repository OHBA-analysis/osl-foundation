def update_history(old_history, new_history):
    if old_history is None:
        old_history = {}

    for k, v in new_history.items():
        if k in old_history:
            old_history[k].extend(v)
        else:
            old_history[k] = v

    return old_history
