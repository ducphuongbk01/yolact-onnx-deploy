import onnxruntime
import cv2
import numpy as np
import time

COCO_CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

colors = [
            [56, 0, 255],
            [226, 255, 0],
            [0, 94, 255],
            [0, 37, 255],
            [0, 255, 94],
            [255, 226, 0],
            [0, 18, 255],
            [255, 151, 0],
            [170, 0, 255],
            [0, 255, 56],
            [255, 0, 75],
            [0, 75, 255],
            [0, 255, 169],
            [255, 0, 207],
            [75, 255, 0],
            [207, 0, 255],
            [37, 0, 255],
            [0, 207, 255],
            [94, 0, 255],
            [0, 255, 113],
            [255, 18, 0],
            [255, 0, 56],
            [18, 0, 255],
            [0, 255, 226],
            [170, 255, 0],
            [255, 0, 245],
            [151, 255, 0],
            [132, 255, 0],
            [75, 0, 255],
            [151, 0, 255],
            [0, 151, 255],
            [132, 0, 255],
            [0, 255, 245],
            [255, 132, 0],
            [226, 0, 255],
            [255, 37, 0],
            [207, 255, 0],
            [0, 255, 207],
            [94, 255, 0],
            [0, 226, 255],
            [56, 255, 0],
            [255, 94, 0],
            [255, 113, 0],
            [0, 132, 255],
            [255, 0, 132],
            [255, 170, 0],
            [255, 0, 188],
            [113, 255, 0],
            [245, 0, 255],
            [113, 0, 255],
            [255, 188, 0],
            [0, 113, 255],
            [255, 0, 0],
            [0, 56, 255],
            [255, 0, 113],
            [0, 255, 188],
            [255, 0, 94],
            [255, 0, 18],
            [18, 255, 0],
            [0, 255, 132],
            [0, 188, 255],
            [0, 245, 255],
            [0, 169, 255],
            [37, 255, 0],
            [255, 0, 151],
            [188, 0, 255],
            [0, 255, 37],
            [0, 255, 0],
            [255, 0, 170],
            [255, 0, 37],
            [255, 75, 0],
            [0, 0, 255],
            [255, 207, 0],
            [255, 0, 226],
            [255, 245, 0],
            [188, 255, 0],
            [0, 255, 18],
            [0, 255, 75],
            [0, 255, 151],
            [255, 56, 0],
            [245, 255, 0],
    ]

class Yolact(object):

    def __init__(self, model_path, conf_threshold=0.4, nms_threshold=0.3, mask_threshold=0.5, top_k=10):

        self.sess = onnxruntime.InferenceSession(model_path)
        self.target_size = 550

        self.MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32).reshape(1, 1, 3)
        self.STD = np.array([57.38, 57.12, 58.40], dtype=np.float32).reshape(1, 1, 3)
        self.variances = [0.1, 0.2]
        
        loc_name    = self.sess.get_outputs()[0].name
        conf_name   = self.sess.get_outputs()[1].name
        mask_name   = self.sess.get_outputs()[2].name
        priors_name = self.sess.get_outputs()[3].name
        proto_name  = self.sess.get_outputs()[4].name
        
        self.names = [loc_name, conf_name, mask_name, priors_name, proto_name]
        self.input_name  = self.sess.get_inputs()[0].name

        self.nms_threshold = nms_threshold
        self.confidence_threshold = conf_threshold
        self.keep_top_k = top_k
        self.mask_threshold = mask_threshold


    def decode(self, loc, priors, img_w, img_h):
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1]),
            ),
            1
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        boxes[:, 0], boxes[:, 2] = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], img_h)
        boxes[:, 1], boxes[:, 3] = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], img_w)

        return boxes
    

    def sanitize_coordinates(self, _x1, _x2, img_size:int):
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size
        x1 = np.min([_x1, _x2], axis=0)
        x2 = np.max([_x1, _x2], axis=0)
        x1 = np.clip(x1, a_min=0, a_max=img_size)
        x2 = np.clip(x2, a_min=0, a_max=img_size)
        return x1, x2
    

    def mask_map_bbox(self, box, mask):
        mask_box = np.zeros_like(mask)
        mask_box[box[1]:box[3], box[0]:box[2]] = 1
        mask_output = mask & mask_box
        return mask_output
    

    def parse_image(self, img):
        img_input = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_input = cv2.resize(img_input, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img_input = (img_input - self.MEANS) / self.STD
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.array([img_input])
        return img_input
        

    def __call__(self, img):
        img_w, img_h, img_c = img.shape

        img_input = self.parse_image(img)

        loc_data, conf_preds, mask_data, priors_data, proto_data = self.sess.run(self.names, {self.input_name: img_input})

        cur_scores = conf_preds[:, 1:]
        num_class = cur_scores.shape[1]
        classid = np.argmax(cur_scores, axis=1)

        conf_scores = cur_scores[range(cur_scores.shape[0]), classid]

        # filte by confidence_threshold
        keep = conf_scores > self.confidence_threshold
        conf_scores = conf_scores[keep]
        classid = classid[keep]
        loc_data = loc_data[keep, :]
        prior_data = priors_data[keep, :]
        boxes = self.decode(loc_data, prior_data, img_w, img_h)
        masks = mask_data[keep, :]

        output_bboxes = []
        output_masks = []
        output_conf = []
        output_classid = []

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf_scores.tolist(), self.confidence_threshold, self.nms_threshold , top_k=self.keep_top_k)
        for i in indices:
            idx = i
            top, left, bottom, right = boxes[idx, :].astype(np.int32).tolist()

            output_bboxes.append((top, left, bottom, right))

            output_classid.append(classid[idx])
            output_conf.append(conf_scores[idx])

            # generate mask
            mask = proto_data @ masks[idx, :].reshape(-1,1)
            mask = 1 / (1 + np.exp(-mask))  ###sigmoid

            # Scale masks up to the full image
            mask = cv2.resize(mask.squeeze(), (img_h, img_w), interpolation=cv2.INTER_LINEAR)
            mask = mask > self.mask_threshold
            mask = self.mask_map_bbox((top, left, bottom, right), mask)

            output_masks.append(mask)

        return output_bboxes, output_masks, output_conf, output_classid 


if __name__ == "__main__":
    net = Yolact("convert-onnx/models/yolact_resnet50_54_800000.onnx", conf_threshold=0.5, nms_threshold=0.4, top_k=100)
    img = cv2.imread("test_img/example_02.jpg")

    t1 = time.time()
    output_bboxes, output_masks, output_conf, output_classid  = net(img)
    print(f"Inferent time: {time.time()-t1}")

    k = 0
    for bbox, classid, mask in zip(output_bboxes, output_classid, output_masks):
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[classid+1], 3)
        img = cv2.putText(img, f"{COCO_CLASSES[classid+1]}", (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        img[mask] = img[mask] * 0.5 + np.array(colors[classid+1]) * 0.5

        cv2.imwrite(f"mask_results/{k}.jpg", (mask*255).astype(np.uint8))
        k+=1
    
    img = img.astype(np.uint8)

    cv2.imwrite("result.jpg", img)

    print(output_classid)
    print(img.shape)
    print(output_masks[0].shape)
