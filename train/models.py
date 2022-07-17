import torchvision.models as models
import torch.nn as nn

def get_model_from_name(name, pretrained=True):
    available_models = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'vgg19_bn': models.vgg19_bn,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'mobilenet_v2': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2
    }

    if name in available_models.keys():
        model = available_models[name] 
        model = model(pretrained) 
        model = replace_classifier_layer(model, 101)    
        return model
    else:
        raise ValueError(
            'Invalid or unsupported model. Options are ' + ', '.join(available_models.keys()))

def freeze_param(model, layer):
    count = 0
    for name, param in model.named_parameters():
        count +=1
        if count < layer:
            param.requires_grad = False

def print_param_model(model):
	total_params = sum(p.numel() for p in model.parameters())
	print('{0:,} total parameters.'.format(total_params))
	total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
	print('{0:,} training parameters.'.format(total_trainable_params))	

def replace_classifier_layer(model, num_classes):
    if isinstance(model, (models.ResNet)):
        final_layer_input_size = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(final_layer_input_size, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, num_classes), nn.Softmax(dim=1))
    elif isinstance(model, models.VGG):
        final_layer_input_size = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(final_layer_input_size, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, num_classes), nn.Softmax(dim=1))
    elif isinstance(model, models.DenseNet):
        final_layer_input_size = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(final_layer_input_size, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, num_classes), nn.Softmax(dim=1))
    elif isinstance(model, models.MobileNetV2):
        final_layer_input_size = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(final_layer_input_size, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, num_classes), nn.Softmax(dim=1))
    else:
        final_layer_input_size = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(final_layer_input_size, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes), nn.Softmax(dim=1))
    #freeze_param(model, 400)
    #print_param_model(model)
    return model
