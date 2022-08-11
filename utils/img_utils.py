import matplotlib.pyplot as plt

def ds_label_map(ds):
    label_map = [None] * ds.num_label

    for i in range(ds.num_label):
        class_idx = ds.label_list[i]
        class_name = ds.classes[class_idx]
        label_map[i] = class_name
    
    return label_map

def view_sample_img(dloader, pred_func, device, label_map=None):
    if label_map is None:
        label_map = ds_label_map(dloader.dataset)
    
    batch_load = iter(dloader)
    img, label = batch_load.next()
    img_gpu = img.to(device)
    pred_label = pred_func(img_gpu)

    for i in range(9):
        plt.subplot(3, 3, i+1)
        rgb_image = img[i].permute(1,2,0)
        plt.imshow(rgb_image.numpy())
        plt.axis('off')
        pred_label_idx = pred_label[i]
        act_label_idx = label[i]
        pred_class_name = label_map[pred_label_idx]
        act_class_name = label_map[act_label_idx]

        plt.title(f'Pred: {pred_class_name}, Act: {act_class_name}')
    plt.show()