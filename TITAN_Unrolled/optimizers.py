

def build_optimizer(model,optimizer_cfg):
        return optimizer_cfg["class"](model.parameters(),**optimizer_cfg["args"])

def build_scheduler(optimizer,scheduler_cfg):
    return scheduler_cfg["class"](optimizer,**scheduler_cfg["args"]) 