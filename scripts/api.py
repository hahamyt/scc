from http import HTTPStatus
import torch
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
from easydict import EasyDict as edict
import cv2
from fastapi.responses import StreamingResponse
from io import BytesIO
import torchvision.transforms as T

from isegm.engine.trainer import load_weights
from isegm.model.is_plainvit_model import PlainVitModel
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

model = None
app = FastAPI(title='SimpleClick API')

toTensor = T.Compose([
    T.ToTensor(),
])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
im_sz = 448
# base
ckpt_path = '../experiments/gnnvit/vitB_comb_gnn/003_gcn_only_pos/checkpoints/690.pth'
# # large
# ckpt_path = '../experiments/gnnvit/vitB_comb_gnn/003_gcn_only_pos/checkpoints/690.pth'

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.on_event('startup')
async def load_model():
    global model
    if model is None:
        model_cfg = edict()
        model_cfg.crop_size = (im_sz, im_sz)
        model_cfg.num_max_points = 24
        model_cfg.use_fp16 = False
        model_cfg.with_prev_mask = True

        backbone_params = dict(
            img_size=model_cfg.crop_size,
            patch_size=(16, 16),
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            use_gcn=True,
        )

        neck_params = dict(
            in_dim=768,
            out_dims=[128, 256, 512, 1024],
        )

        head_params = dict(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            num_classes=1,
            loss_decode=CrossEntropyLoss(),
            align_corners=False,
            upsample='x1',
            channels=256,
        )

        model = PlainVitModel(
            use_disks=True,
            norm_radius=5,
            with_prev_mask=model_cfg.with_prev_mask,
            backbone_params=backbone_params,
            neck_params=neck_params,
            head_params=head_params,
            random_split=False,
        ).to(device)
        # 加载权重
        load_weights(model, ckpt_path)


@app.get('/')
def root():
    '''Health check.'''
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {},
    }
    return response

def resolve_pts(pos_clicks, neg_clicks):
    # resolve points
    if len(pos_clicks) != 0:
        try:
            pos = np.array(
                [[int(p.split(',')[0]), int(p.split(',')[1])] for p in pos_clicks.split(';')])
        except:
            pos = np.array(pos_clicks)
    else:
        return
    if len(neg_clicks) != 0:
        try:
            neg = np.array(
                [[int(p.split(',')[0]), int(p.split(',')[1])] for p in neg_clicks.split(';')])
        except:
            neg = np.array(neg_clicks)
    else:
        neg = None
    return {'pos': pos, 'neg': neg}

# 兼容RITM的格式
def prepare_points(point_coords, point_labels):
    pos = point_coords[point_labels == 1]
    pos[:, [0, 1]] = pos[:, [1, 0]]
    neg = point_coords[point_labels == 0]
    neg[:, [0, 1]] = neg[:, [1, 0]]
    num_clicks = max(pos.shape[0], neg.shape[0])
    points = np.ones([2 * num_clicks, 3]) * -1
    points[:pos.shape[0], 0:2] = pos
    points[:pos.shape[0], 2] = 1
    points[num_clicks: neg.shape[0] + num_clicks, 0:2] = neg
    points[num_clicks: neg.shape[0] + num_clicks, 2] = 0
    return torch.from_numpy(points[np.newaxis,  :, :])

@app.post('/simpleclick/')
async def predict(pos_clicks: str = Form(...),
                  neg_clicks: str = Form(...),
                  img: UploadFile = File(...),
                  prev_mask: UploadFile = File(...)
                  ):
    global model
    pts = resolve_pts(pos_clicks, neg_clicks)
    input_point = pts['pos']
    input_label = np.ones(pts['pos'].shape[0])

    if pts['neg'] is not None:
        for x in pts['neg']:
            input_point = np.vstack([input_point, x[np.newaxis, :]])
            input_label = np.hstack([input_label, (0,)])

    points = prepare_points(input_point, input_label)

    contents = await img.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, [im_sz, im_sz])
    try:
        contents = await prev_mask.read()
        nparr = np.frombuffer(contents, np.uint8)
        prev_mask = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        prev_mask = cv2.resize(prev_mask, [im_sz, im_sz])[:, :, :1]
    except:
        prev_mask = np.zeros([im_sz, im_sz], dtype=np.float32)

    image = toTensor(img).unsqueeze(0)
    prev_mask = toTensor(prev_mask).unsqueeze(0)

    input = torch.cat((image, prev_mask), dim=1)
    with torch.no_grad():
        outputs = model(input.to(device), points.to(device))
        pred_probs = torch.sigmoid(outputs['instances']).squeeze().cpu().numpy() * 255

    _, encoded_img = cv2.imencode('.png', pred_probs)
    img_bytes = BytesIO(encoded_img.tobytes())
    response = StreamingResponse(img_bytes, media_type="image/png")
    return response


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=6006, reload=True)