import sys

import numpy as np
import tensorflow as tf

model_dir, train_file, test_file = sys.argv[1:4]


def load(name):
    it = tf.lite.Interpreter(model_path=f"{model_dir}/{name}.tflite")
    it.allocate_tensors()
    return it


def run(it, inputs):
    for d, x in zip(it.get_input_details(), inputs):
        it.set_tensor(d["index"], np.asarray(x, dtype=np.float32).reshape(d["shape"]))
    it.invoke()
    return [it.get_tensor(d["index"]).copy() for d in it.get_output_details()]


def read(path):
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    y = np.array([int(float(r[0])) for r in rows])
    x = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)
    return y, x


init, train, opt, inf = load("initialize"), load("train_head"), load("optimizer"), load("inference")
n_params = len(train.get_input_details()) - 2
batch = train.get_input_details()[0]["shape"][0]
n_opt_state = len(opt.get_input_details()) - 2 * n_params
assert n_opt_state == 0, "optimizer has state (e.g. Adam) - the Java client cannot handle that, use SGD"
print(f"parameter tensors: {n_params}, train batch size: {batch}")

ytr, xtr = read(train_file)
yte, xte = read(test_file)
xb = xtr[ytr == 0]  # the autoencoder only trains on benign rows (labelled 0)
print(f"train rows: {len(xtr)} (benign {len(xb)}), test rows: {len(xte)} (malicious {int((yte == 1).sum())})")
assert len(xb) >= batch, "fewer benign rows than the batch size"

params = run(init, [0.0])
rng = np.random.default_rng(0)


def errors(x):
    return np.array([float(np.mean((run(inf, [r] + params)[0].ravel() - r) ** 2)) for r in x])


print(f"benign test MSE before training: {errors(xte[yte == 0]).mean():.5f}")
for epoch in range(30):
    idx = rng.permutation(len(xb))
    losses = []
    for s in range(0, len(idx) - batch + 1, batch):
        b = xb[idx[s : s + batch]]
        out = run(train, [b, b] + params)  # inputs: bottleneck, target, *params
        losses.append(float(out[0]))
        params = run(opt, params + out[1:])  # inputs: *params, *grads -> new params
    if epoch % 5 == 0 or epoch == 29:
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.5f}")

e = errors(xte)
thr = np.percentile(e[yte == 0], 85)
pred = e > thr
print(f"benign test MSE after training: {e[yte == 0].mean():.5f}, threshold (85th pct): {thr:.5f}")
if (yte == 1).any():
    print(f"malicious test MSE: {e[yte == 1].mean():.5f}")
print(f"accuracy: {(pred == (yte == 1)).mean():.3f}")
print("OK: model trains and evaluates in the TFLite format")
