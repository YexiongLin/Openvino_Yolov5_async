from common import *
import logging as log
from openvino.inference_engine import IECore, IENetwork, IEPlugin
import threading
from threading import Lock
import time
import queue


def build_argparser():
    parser = ArgumentParser(prog="demo_openvino.py")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on; CPU, GPU, HDDL, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)",
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default="./yolov5s.xml",
        help="model.xml path",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="./zidane.jpg",
        help="image path",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.3, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.4, help="IOU threshold for NMS"
    )
    parser.add_argument("--cv", action="store_true",
                        help="use PIL or opencv image")

    return parser


num_requests = 16
orig_imgs = [i for i in range(num_requests)]
lock = Lock()
infer_request = None
requests_list = queue.Queue()
start_times = [0 for i in range(num_requests)]
start = 0
end = 0
for i in range(num_requests):
    requests_list.put(i)


def callback(status, id):
    global requests_list, start, end
    # infer_request[i].wait()
    out_blob = next(
        iter(infer_request[id].output_blobs))
    res = infer_request[id].output_blobs[out_blob]
    output = res.buffer

    # Processing output blob
    log.info("Processing output blob")

    detections = non_max_suppression(
        output, conf_thres=args.conf_thres, iou_thres=args.iou_thres, agnostic=False
    )

    display(
        detections[0], orig_imgs[id], input_size=args.img_size, text_bg_alpha=0.6
    )
    end = time.time()
    print(f"delay time: {(end - start_times[id])}")
    print(f"frame rate: {1/(end - start)}")
    start = time.time()
    requests_list.put(id)


def main(args):
    global orig_imgs, infer_request, requests_list, start_times
    model_bin = os.path.splitext(args.model_xml)[0] + ".bin"

    time.sleep(1)
    ie = IECore()
    net = ie.read_network(model=args.model_xml, weights=model_bin)
    exec_net = ie.load_network(
        network=net, device_name=args.device, num_requests=num_requests)

    # create one inference request for asynchronous execution
    infer_request = exec_net.requests

    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        #start = time.time()
        ret, img = cap.read()
        if ret and requests_list.not_empty:
            orig_img = img.copy()
            i = requests_list.get()
            lock.acquire()
            orig_imgs[i] = orig_img
            lock.release()
            img_in = preprocess_cv_img(orig_img, args.img_size, args.img_size)
            inputs = infer_request[i].input_blobs
            inputs['images'].buffer[:] = img_in
            now = time.time()
            start_times[i] = now
            infer_request[i].set_completion_callback(
                py_callback=callback, py_data=i)
            infer_request[i].async_infer()
        #end = time.time()
        #print(f"Processing time: {(end - start)}")

    cv2.destroyAllWindows()
    del net
    del exec_net
    del ie


if __name__ == "__main__":
    args = build_argparser().parse_args()

    sys.exit(main(args) or 0)
