import time

def eval_sto_simple(model,dataset):
    start_time = time.time()
    for x_batch_val, y_batch_val in dataset:
        prediction = model(x_batch_val, training=False)
        measures = model.compute_metrics(x=None,y=y_batch_val,y_pred=prediction, sample_weight=None)
    end_time = time.time()

    for k,v in measures.items():
        measures[k] = float(v)

    return measures, end_time-start_time

def eval_sto_inversion(model, dataset, transformerY=None):
    start_time = time.time()
    for x_batch_val, y_batch_val in dataset:
        prediction = model(x_batch_val, training=False)
        if transformerY is not None:
            prediction = transformerY.inverse_transform(prediction)
            y_batch_val = transformerY.inverse_transform(y_batch_val)
        measures = model.compute_metrics(x=None,y=y_batch_val,y_pred=prediction, sample_weight=None)
    end_time = time.time()

    for k,v in measures.items():
        measures[k] = float(v)

    return measures, end_time-start_time

def eval_sto_global(name_eval,model,dataset,transformerY=None):
    if(name_eval=="simple"):
        return eval_sto_simple(model,dataset)
    elif(name_eval=="inversion"):
        return eval_sto_inversion(model,dataset,transformerY)
