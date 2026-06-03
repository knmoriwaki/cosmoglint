import sys
import h5py
import numpy as np


def periodic_delta(dx, boxsize):
    return dx - boxsize * np.rint(dx / boxsize)


for lh_number in range(1000):
    base_dir = f"../data/halo/CAMELS/IllustrisTNG/LH/LH_{lh_number}"
    input_file = f"{base_dir}/groups_032.hdf5"
    output_file = f"{base_dir}/my_groups_032.hdf5"

    with h5py.File(input_file, "r") as fin, h5py.File(output_file, "w") as fout:
        for key, val in fin.attrs.items():
            fout.attrs[key] = val

        for name in fin.keys():
            fin.copy(name, fout, name=name)


    chunk_size=1000000
    with h5py.File(output_file, "r+") as f:
        group_pos = f["Group/GroupPos"][:]       # (Ngroup, 3)
        group_vel = f["Group/GroupVel"][:]       # (Ngroup, 3)
        group_len = f["Group/GroupNsubs"][:]

        sub_pos_ds = f["Subhalo/SubhaloPos"]
        sub_vel_ds = f["Subhalo/SubhaloVel"]

        n_sub = sub_pos_ds.shape[0]

        # Header/BoxSize attribute を取得
        if "Header" in f and "BoxSize" in f["Header"].attrs:
            boxsize = f["Header"].attrs["BoxSize"]
        else:
            raise KeyError("Header.attrs['BoxSize'] が見つかりません。")

        boxsize = float(np.asarray(boxsize))

        subhalo_group = f["Subhalo"]

        for name in ["SubhaloDist", "SubhaloVrad", "SubhaloVtan"]:
            if name in subhalo_group:
                del subhalo_group[name]

        d_dist = subhalo_group.create_dataset(
            "SubhaloDist",
            shape=(n_sub,),
            dtype="f4",
            chunks=True,
            compression="gzip",
        )
        d_vrad = subhalo_group.create_dataset(
            "SubhaloVrad",
            shape=(n_sub,),
            dtype="f4",
            chunks=True,
            compression="gzip",
        )
        d_vtan = subhalo_group.create_dataset(
            "SubhaloVtan",
            shape=(n_sub,),
            dtype="f4",
            chunks=True,
            compression="gzip",
        )

        d_dist.attrs["Description"] = "Distance from host Group center with periodic boundary correction"
        d_vrad.attrs["Description"] = "Radial velocity relative to host Group velocity"
        d_vtan.attrs["Description"] = "Tangential velocity relative to host Group velocity"

        # 各 Group が担当する Subhalo index の終端
        # 例: GroupLen = [2, 3, 1] -> group_ends = [2, 5, 6]
        group_ends = np.cumsum(group_len)

        for start in range(0, n_sub, chunk_size):
            end = min(start + chunk_size, n_sub)

            sub_indices = np.arange(start, end)

            # 各 Subhalo がどの Group に属するか
            parent_group_index = np.searchsorted(
                group_ends,
                sub_indices,
                side="right",
            )

            sub_pos = sub_pos_ds[start:end]
            sub_vel = sub_vel_ds[start:end]

            host_pos = group_pos[parent_group_index]
            host_vel = group_vel[parent_group_index]

            dx = sub_pos - host_pos
            dx = periodic_delta(dx, boxsize)

            dv = sub_vel - host_vel

            dist = np.linalg.norm(dx, axis=1)

            # dist=0 の場合のゼロ割りを避ける
            rhat = np.zeros_like(dx)
            nonzero = dist > 0
            rhat[nonzero] = dx[nonzero] / dist[nonzero, None]

            vrad = np.sum(dv * rhat, axis=1)

            v2 = np.sum(dv**2, axis=1)
            vtan2 = v2 - vrad**2
            vtan2 = np.maximum(vtan2, 0.0)
            vtan = np.sqrt(vtan2)

            d_dist[start:end] = dist.astype(np.float32)
            d_vrad[start:end] = vrad.astype(np.float32)
            d_vtan[start:end] = vtan.astype(np.float32)

            print(f"{lh_number}: processed {end} / {n_sub}")
