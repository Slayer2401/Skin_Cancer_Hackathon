import h5py
import json

def fix_layer(layer):
    if not isinstance(layer, dict): return
    if 'class_name' in layer and 'config' in layer:
        # Fix InputLayer
        if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
        
        # Fix DTypePolicy
        if 'dtype' in layer['config'] and isinstance(layer['config']['dtype'], dict):
            layer['config']['dtype'] = layer['config']['dtype'].get('config', {}).get('name', 'float32')
            
        # Fix preprocessing layers
        if layer['class_name'] in ['RandomFlip', 'RandomRotation', 'RandomZoom', 'RandomTranslation', 'RandomContrast', 'RandomBrightness', 'RandomCrop', 'Resizing', 'Rescaling']:
            for key in ['data_format', 'value_range']:
                if key in layer['config']:
                    del layer['config'][key]
                    
        # Recurse if it's a nested model
        if 'layers' in layer['config']:
            for sub_layer in layer['config']['layers']:
                fix_layer(sub_layer)

try:
    with h5py.File('skin_cancer_model_v2_perfect.h5', 'r+') as f:
        model_config = json.loads(f.attrs.get('model_config', '{}'))
        if 'config' in model_config and 'layers' in model_config['config']:
            for layer in model_config['config']['layers']:
                fix_layer(layer)
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        print("Model file patched successfully recursively for Keras 2!")
except Exception as e:
    import traceback
    traceback.print_exc()
