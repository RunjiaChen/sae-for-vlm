"""Deterministic probing-set loader for the ImageNet-1K val set.

Source: HF `clip-benchmark/wds_imagenet1k`, split `test/` = the 50k ILSVRC2012 val
images as webdataset tar shards 0.tar..6.tar. We never shuffle: probing-set row i is
defined by the manifest (shards in numeric order, members in stored tar order), and
every pipeline stage indexes images the same way.

The shards are extracted once per job to fast node-local disk; a manifest.json in the
workspace records the canonical order so extraction is reproducible.
"""

import json
import os
import shutil
import tarfile

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def shard_paths(wds_dir):
    shards = sorted(
        (f for f in os.listdir(wds_dir) if f.endswith('.tar')),
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    return [os.path.join(wds_dir, f) for f in shards]


def build_manifest(wds_dir, manifest_path):
    """List of [shard_basename, member_name] in canonical order; cached on disk."""
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    entries = []
    for shard in shard_paths(wds_dir):
        with tarfile.open(shard) as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(IMG_EXTS):
                    entries.append([os.path.basename(shard), m.name])
    tmp = manifest_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(entries, f)
    os.replace(tmp, manifest_path)
    return entries


def extract_images(wds_dir, manifest, out_dir):
    """Extract to out_dir/{i:06d}{ext} following manifest order.

    A dir without the done-flag may hold truncated files from a killed attempt
    (a requeued job can land on the same node and reuse the path), so it is wiped
    and rebuilt; extraction only takes minutes.
    """
    done_flag = os.path.join(out_dir, '.extract_done')
    if os.path.exists(done_flag):
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    by_shard = {}
    for i, (shard, member) in enumerate(manifest):
        by_shard.setdefault(shard, []).append((i, member))
    for shard, items in by_shard.items():
        wanted = {member: i for i, member in items}
        with tarfile.open(os.path.join(wds_dir, shard)) as tf:
            for m in tf:
                if m.name in wanted:
                    i = wanted[m.name]
                    ext = os.path.splitext(m.name)[1].lower()
                    dst = os.path.join(out_dir, f'{i:06d}{ext}')
                    with tf.extractfile(m) as src, open(dst + '.tmp', 'wb') as out:
                        out.write(src.read())
                    os.replace(dst + '.tmp', dst)
    with open(done_flag, 'w') as f:
        f.write('ok')


class ProbingDataset(Dataset):
    """Map-style dataset over the extracted probing images. Item i -> PIL RGB image."""

    def __init__(self, img_dir, manifest):
        self.img_dir = img_dir
        self.paths = [
            os.path.join(img_dir, f'{i:06d}{os.path.splitext(m)[1].lower()}')
            for i, (_, m) in enumerate(manifest)
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert('RGB')
