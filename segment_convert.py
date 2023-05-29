import json
import random
def uniqueid():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

unique_sequence = uniqueid()

def extract_ls_format(images):
    """
    images format:
    images = [
        {
            "url": image1's url,
            "width": width of image,
            "height": heigt of image,
            "annotations": [
                {
                "segmentation":[],
                "category_name": category name
                },
                {
                "segmentation":[],
                "category_name": category name
                },
                {
                "segmentation":[],
                "category_name": category name
                },...
        },
        {
            "url": image2's url,
            "width": width of image,
            "height": heigt of image,
            "annotations": [
                {
                "segmentation":[],
                "category_name": category name
        },...
    ]
    }
    ]

    """
    annotation = []
    for idx, img in enumerate(images):

        img_width = img['width']
        img_height = img['height']

        results = {"annotations": [{}], "data": {
                "image": img['url']}}
        
        results['annotations'][0]["id"] = idx
        results['annotations'][0]['completed_by'] = 1

        result = []

        for seg in img['annotations']:
            points = []
            width = img_width
            height = img_height
            segment = seg['segmentation']
            for i in range(0, len(segment), 2):
                x = (segment[i]/width)*100
                y = (segment[i+1]/height)*100
                points.append([x,y])



            cat_name = seg['category_name']

            result.append(
                {
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {"points": points,
                            "closed": True,
                            "polygonlabels": [cat_name]
                            },
                    "id": next(unique_sequence),
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "origin": "manual"
                })
            
        results['annotations'][0]['result'] = result
        annotation.append(results)
    
    return annotation