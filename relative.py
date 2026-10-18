import numpy as np, yaml
import glob, os
import cv2

DEFAULT_OPTION = "B"

def pose_to_matrix(p):
    x, y, z, roll, yaw, pitch = p
    cr, sr = np.cos(np.radians(roll)),  np.sin(np.radians(roll))
    cy, sy = np.cos(np.radians(yaw)),   np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    M = np.eye(4)
    M[:3, :3] = [[cp*cy, cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr],
                 [cp*sy, sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr],
                 [sp,    -cp*sr,           cp*cr]]
    M[:3, 3] = [x, y, z]
    return M

# Dataset camera frame -> OpenCV frame (x right, y down, z forward)
P_OPTIONS = {
    "A": np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float),  # cam is x fwd, y LEFT,  z up
    "B": np.array([[0,  1, 0], [0, 0, -1], [1, 0, 0]], float),  # cam is x fwd, y RIGHT, z up
}

def relative_pose(d1, d2, cam="cam1_left", option="B"):
    """Returns R, t with X_cv2 = R @ X_cv1 + t (OpenCV frames)."""
    E1 = np.array(d1[cam]["extrinsic"])
    E2 = np.array(d2[cam]["extrinsic"])
    T_w_l1 = pose_to_matrix(d1["lidar_pose"])
    T_w_l2 = pose_to_matrix(d2["lidar_pose"])
    T_c2_c1 = E2 @ np.linalg.inv(T_w_l2) @ T_w_l1 @ np.linalg.inv(E1)
    P4 = np.eye(4)
    P4[:3, :3] = P_OPTIONS[option]
    T = P4 @ T_c2_c1 @ np.linalg.inv(P4)
    return T[:3, :3], T[:3, 3]

def load_lidar(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)

def fit_ground_plane(pts, iters=500, thresh=0.10, seed=0):
    """RANSAC plane in LiDAR frame. Returns unit normal n (pointing up) and b with n·X + b = 0."""
    rng = np.random.default_rng(seed)
    # candidates: nearby points in the lowest part of the cloud
    m = (np.abs(pts[:, 0]) < 40) & (np.abs(pts[:, 1]) < 15) & (pts[:, 2] < np.percentile(pts[:, 2], 40))
    cand = pts[m]
    best, best_n = None, 0
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < 0.9:          # ground should be close to horizontal
            continue
        inl = np.abs((cand - s[0]) @ n) < thresh
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    P = cand[best]              # refine on inliers with SVD
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    if n[2] < 0:
        n = -n
    return n, -n @ c, best_n, len(cand)

def road_plane_in_cam(d1, bin1, cam="cam1_left", option="B"):
    """n_down, d with n_down · X = d in Car 1's OpenCV camera frame (d = camera height)."""
    n_l, b, n_in, n_cand = fit_ground_plane(load_lidar(bin1))
    E = np.array(d1[cam]["extrinsic"])
    c_l = -E[:3, :3].T @ E[:3, 3]            # camera center in LiDAR frame
    d = n_l @ c_l + b                        # signed height above the plane
    n_cv_up = P_OPTIONS[option] @ E[:3, :3] @ n_l
    return -n_cv_up, d, n_in, n_cand         # OpenCV y is down, so the normal toward the road is -up

def K_of(d, cam="cam1_left"):
    return np.array(d[cam]["intrinsic"])

def road_homography(K1, K2, R, t, n, d):
    """H_2_from_1: maps pixels of Car 1's image to Car 2's, for points on the plane n·X1 = d."""
    H = K2 @ (R + np.outer(t, n) / d) @ np.linalg.inv(K1)
    return H / H[2, 2]

def warp_overlay(img1, img2, H, alpha=0.5):
    warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    return cv2.addWeighted(img2, alpha, warped, 1 - alpha, 0)
def skew(t):
    t = np.asarray(t).ravel()
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])

def fundamental(K1, K2, R, t):
    """x2^T F x1 = 0, where X2 = R X1 + t (OpenCV frames)."""
    F = np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)
    return F / np.linalg.norm(F)

def lidar_to_cv(d, pts_l, cam="cam1_left", option="B"):
    """LiDAR-frame points (N,3) -> this car's OpenCV camera frame (N,3)."""
    E = np.array(d[cam]["extrinsic"])
    Xc = pts_l @ E[:3, :3].T + E[:3, 3]
    return Xc @ P_OPTIONS[option].T

def project(K, X):
    x = X @ K.T
    return x[:, :2] / x[:, 2:3]

def self_check(d1, d2, bin1, option="B", cam="cam1_left"):
    """Image-free check that H and F agree with the full 3D transform."""
    R, t = relative_pose(d1, d2, cam, option)
    n, d, *_ = road_plane_in_cam(d1, bin1, cam, option)
    K1, K2 = K_of(d1, cam), K_of(d2, cam)
    H = road_homography(K1, K2, R, t, n, d)
    F = fundamental(K1, K2, R, t)

    X1 = lidar_to_cv(d1, load_lidar(bin1), cam, option)
    X2 = X1 @ R.T + t
    ok = (X1[:, 2] > 2) & (X2[:, 2] > 2)            # in front of both cameras
    X1, X2 = X1[ok], X2[ok]

    # New addition
    if ok.sum() == 0:
        print("  no points in front of both cameras (cars probably face opposite directions)")
        return

    # F: should hold for ANY 3D point seen by both cameras
    p1, p2 = project(K1, X1), project(K2, X2)
    x1h, x2h = np.c_[p1, np.ones(len(p1))], np.c_[p2, np.ones(len(p2))]
    l2 = x1h @ F.T                                   # epipolar lines in image 2
    dist = np.abs((l2 * x2h).sum(1)) / np.linalg.norm(l2[:, :2], axis=1)
    print(f"  F: {len(p1)} points, max point-to-line distance {dist.max():.2e} px")

    # H: only for points ON the plane, so snap near-ground points onto it exactly
    g = np.abs(X1 @ n - d) < 0.15
    Xp = X1[g] + np.outer(d - X1[g] @ n, n)
    Xp2 = Xp @ R.T + t
    ok2 = (Xp[:, 2] > 2) & (Xp2[:, 2] > 2)
    Xp, Xp2 = Xp[ok2], Xp2[ok2]
    q1, q2 = project(K1, Xp), project(K2, Xp2)
    qh = np.c_[q1, np.ones(len(q1))] @ H.T
    qh = qh[:, :2] / qh[:, 2:3]
    err = np.linalg.norm(qh - q2, axis=1)
    print(f"  H: {len(q1)} plane points, max error {err.max():.2e} px")

def find_pairs(scene, collab="1", ego="2", option="B"):
    """Ego sees collaborator ahead: ego camera is 5-80 m behind collab's, heading diff < 30 deg."""
    ld = lambda p: yaml.load(open(p), Loader=yaml.CSafeLoader)
    out = []
    for f in sorted(glob.glob(f"{scene}/{collab}/*.yaml")):
        fr = os.path.basename(f)[:-5]
        g = f"{scene}/{ego}/{fr}.yaml"
        if not os.path.exists(g):
            continue
        a, b = ld(f), ld(g)
        if a.get("infra") or b.get("infra"):
            continue
        R, t = relative_pose(a, b, option=option)
        c2 = -R.T @ t                                  # ego camera in collab's camera frame
        heading = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        dist = np.linalg.norm(c2)
        if -c2[2] > 0 and 5 < dist < 80 and heading < 30:
            out.append((fr, c2.round(1), round(heading, 1)))
    return out

def scan_all(root="dataset"):
    for scene in sorted(glob.glob(f"{root}/*/")):
        scene = scene.rstrip("/")
        for collab, ego in [("1", "2"), ("2", "1")]:
            if not (os.path.isdir(f"{scene}/{collab}") and os.path.isdir(f"{scene}/{ego}")):
                continue
            pairs = find_pairs(scene, collab, ego)
            print(f"{os.path.basename(scene)}  collab={collab} ego={ego}: {len(pairs)} frames")
            for p in pairs[:3]:
                print("    ", p)

def yaw_check(scene, fr, collab="1", ego="2"):
    ld = lambda p: yaml.load(open(p), Loader=yaml.CSafeLoader)
    a = ld(f"{scene}/{collab}/{fr}.yaml")
    b = ld(f"{scene}/{ego}/{fr}.yaml")
    assertRaises, t = relative_pose(a, b)
    h_R = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    dy = (b["lidar_pose"][4] - a["lidar_pose"][4] + 180) % 360 - 180
    print(os.path.basename(scene), fr, "| heading from R:", round(h_R, 1),
          "| yaw diff from poses:", round(abs(dy), 1))               

def diagnose(root="dataset"):
    ld = lambda p: yaml.load(open(p), Loader=yaml.CSafeLoader)
    for scene in sorted(glob.glob(f"{root}/*/")):
        scene = scene.rstrip("/")
        for collab, ego in [("1", "2"), ("2", "1")]:
            if not (os.path.isdir(f"{scene}/{collab}") and os.path.isdir(f"{scene}/{ego}")):
                continue
            rows = []
            for f in sorted(glob.glob(f"{scene}/{collab}/*.yaml")):
                fr = os.path.basename(f)[:-5]
                g = f"{scene}/{ego}/{fr}.yaml"
                if not os.path.exists(g):
                    continue
                a, b = ld(f), ld(g)
                if a.get("infra") or b.get("infra"):
                    continue
                R, t = relative_pose(a, b)
                c2 = -R.T @ t                      # ego camera in collab's camera frame
                heading = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
                rows.append((-c2[2], np.linalg.norm(c2), heading))
            if not rows:
                continue
            r = np.array(rows)
            ahead = r[:, 0] > 0
            dist_ok = (r[:, 1] > 5) & (r[:, 1] < 80)
            head_ok = r[:, 2] < 30
            print(f"{os.path.basename(scene)} collab={collab} ego={ego}: "
                  f"frames={len(r)} ahead={ahead.sum()} dist5-80={dist_ok.sum()} "
                  f"heading<30={head_ok.sum()} ALL={(ahead & dist_ok & head_ok).sum()} "
                  f"| median dist={np.median(r[:, 1]):.0f} m, median heading={np.median(r[:, 2]):.0f} deg")

def F_2_from_1(d1, d2, cam="cam1_left", option=DEFAULT_OPTION):
    R, t = relative_pose(d1, d2, cam, option)
    return fundamental(K_of(d1, cam), K_of(d2, cam), R, t)

def H_2_from_1(d1, d2, bin1, cam="cam1_left", option=DEFAULT_OPTION):
    """Car 1's front image -> Car 2's front image, for points on the road plane.
    d1, d2: parsed yaml dicts of the two cars for the same frame; bin1: Car 1's LiDAR .bin path.
    Returns H, and a dict with K1, K2, R, t, n, d for reuse."""
    R, t = relative_pose(d1, d2, cam, option)
    n, d, _, _ = road_plane_in_cam(d1, bin1, cam, option)
    K1, K2 = K_of(d1, cam), K_of(d2, cam)
    H = road_homography(K1, K2, R, t, n, d)
    return H, dict(K1=K1, K2=K2, R=R, t=t, n=n, d=d)

if __name__ == "__main__":
    SCENE = "dataset/2023-04-07-15-05-15_4_1"
    FRAME = "000050"
    ld = lambda p: yaml.load(open(p), Loader=yaml.CSafeLoader)
    d1 = ld(f"{SCENE}/1/{FRAME}.yaml")
    d2 = ld(f"{SCENE}/2/{FRAME}.yaml")
    bin1 = f"{SCENE}/1/{FRAME}.bin"

    H, info = H_2_from_1(d1, d2, bin1)           # uses DEFAULT_OPTION ("B")
    print("H =\n", H.round(4))
    print("camera height d:", round(info["d"], 3))

    F = F_2_from_1(d1, d2)
    print("F =\n", F)

    # to dataset the other left/right option:
    H_A, _ = H_2_from_1(d1, d2, bin1, option="A")
