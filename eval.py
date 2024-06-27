import tensorflow as tf
import time

def eval_simple(model,x,y,sample_weight=None):
    start_time = time.time()
    y_pred = model(x,training=False)
    end_time = time.time()

    y = tf.convert_to_tensor(y)
    y_pred = tf.convert_to_tensor(y_pred)
    if not sample_weight is None:
        sample_weight = tf.convert_to_tensor(sample_weight)

    measures = model.compute_metrics(x=None,y=y,y_pred=y_pred,sample_weight=sample_weight)
    for k,v in measures.items():
        measures[k] = float(v)
    
    return measures, end_time-start_time

def eval_inversion(model,x,y,transformerY=None,sample_weight=None):
    start_time = time.time()
    y_pred = model(x,training=False)
    end_time = time.time()

    if transformerY is not None:
        y_pred = transformerY.inverse_transform(y_pred)
        y = transformerY.inverse_transform(y)

    y = tf.convert_to_tensor(y)
    y_pred = tf.convert_to_tensor(y_pred)
    if not sample_weight is None:
        sample_weight = tf.convert_to_tensor(sample_weight)

    measures = model.compute_metrics(x=None,y=y,y_pred=y_pred,sample_weight=sample_weight)

    for k,v in measures.items():
        measures[k] = float(v)
    return measures,end_time-start_time

def eval_global(name_eval,model,x,y,transformerY=None,sample_weight=None):
    if(name_eval=="simple"):
        return eval_simple(model,x,y,sample_weight)
    elif(name_eval=="inversion"):
        return eval_inversion(model,x,y,transformerY,sample_weight)