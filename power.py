### Functions for computing anuglar power spectrum
### based on power_box's implementation

import sys
import warnings
try:
    from multiprocessing import cpu_count
    THREADS = cpu_count()
    from pyfftw.interfaces.numpy_fft import fftn as _fftn, ifftn as _ifftn, fftfreq
    def fftn(*args, **kwargs):
        return _fftn(threads=THREADS, *args, **kwargs)
    def ifftn(*args, **kwargs):
        return _ifftn(threads=THREADS, *args, **kwargs)
except ImportError:
    #warnings.warn("You do not have pyFFTW installed. Installing it should give some speed increase.")
    print("Warning: you are not using FFTW.", file=sys.stderr)
    from numpy.fft import fftn, ifftn, fftfreq 
import numpy as np # import of numpy should be done after pyfftw import


def my_fft(X, L=None, b=1.): # b = 2 * np.pi for inverse FT

    dim = len(X.shape)
    N = np.array(X.shape)

    if L is None:
        L = N
    elif np.isscalar(L):
        L = L*np.ones(dim)

    dx = np.array([float(l)/float(n) for l, n in zip(L,N)]) # array of side length of cell
    Vx = np.prod(dx) # volume of cell

    ### compute fft ###
    freq = [fftfreq(n, d=d)*2*np.pi for n, d in zip(N, dx)] ## k = 2pi * f
    ft = Vx * fftn(X) * np.sqrt(1/np.abs(b)) ** dim

    return ft, freq


def angular_average_nd(field, coords, nbins, n=None, log_bins=False, indx_min=0):
    ## can be used for real field only

    dim = len(field.shape)
    if len(coords) != dim:
        raise ValueError("coords should be a list of arrays, one for each dimensiont.")

    coords = np.sqrt(np.sum(np.meshgrid(*([x**2] for x in coords)), axis=0)) # k(i,j) = sqrt( ki*ki + kj*kj )

    ### define bins ###
    # "bins" here includes nbins + 1 components corresponding to the edge values of the k-bin
    # if you want to compute the "k of the bin", then you should take (bins[i]+bins[i+1])/2
    mx = coords.max()
    if not log_bins:
        bins = np.linspace(coords.min(), mx, nbins + 1)
    else:
        mn = coords[coords>0].min()
        bins = np.logspace(np.log10(mn), np.log10(mx), nbins + 1)

    ### set index for each pixel ###
    # label the field with a indx that satisfies bins[indx] <= coords < bins[indx+1]
    # indx takes 0, 1, ..., nbin+1
    indx = np.digitize(coords.flatten(), bins)

    ### compute the number of pixels in each bin ###
    # each component in bincount correaponds to the number of pixels with indx = 0, 1, ..., nbin+1
    # i.e., bincount has nbin + 2 = len(bins) + 1 components
    # the first component of bincount is for values < mn.
    # the last component of bincount is for values > mx, where there should be no such pixels. So bincount[-1] is always 0.
    # we remove these two compontnes by setting [1:-1].
    sumweights = np.bincount(indx, minlength=len(bins)+1)[1:-1] 
    if np.any(sumweights==0):
        warnings.warn("One or more radial bins had no cell within it. Use a smaller nbins.")
    if np.any(sumweights==1):
        print("Warning: one or more radial bins have only one cell within it. This would result in inf in the variance")

    ### compute the mean in each bin ###
    # for each bin, sum up the field values, and then divide by the number of pixels
    mean = np.bincount(indx, weights=np.real(field.flatten()), minlength=len(bins)+1)[1:-1] / sumweights

    ### compute variance ### 
     # for each bin, sum up the variance field values (i.e., (field-average)**2), and then divide by the number of pixels
    average_field = np.concatenate(([0], mean, [0]))[indx]
    var_field = ( field.flatten() - average_field ) ** 2
    var = np.bincount(indx, weights=var_field, minlength=len(bins)+1)[1:-1] / (sumweights-1) 

    return mean[indx_min:], bins[indx_min:], var[indx_min:]

def compute_power(deltax, deltax2=None, boxlength=1., nbins=20, log_bins=False):
    ## compmute power spectrum 
    # Inputs:
    #   deltax: input image 
    #   deltax2: second input image. If this is not None, the cross-power spectrum will be computed
    #   boxlength: float or list of floats. The physical length of the side(s) in real space
    #   nbins: int. number of bins
    #   log_bins: use log-scale bins if True.
    #
    # Outputs: 
    #   Pk: angular-averaged power spectrum. 
    #   k: edge values of the bins. Note there are N+1 values when the length of Pk is N.
    #   var: variance 

    if deltax2 is not None and deltax.shape != deltax2.shape:
        raise ValueError("deltax and deltax2 must have the same shpae!")

    dim = len(deltax.shape)

    if not np.iterable(boxlength):
        boxlength = [boxlength] * dim
    V = np.prod(boxlength)

    # Fourier transformation
    FT, freq = my_fft(deltax, L=boxlength) 

    if deltax2 is not None:
        FT2, _ = my_fft(deltax2, L=boxlength)
    else:
        FT2 = FT

    # compute power spectrum
    P = np.real(FT * np.conj(FT2)) / V
    """ we need to implement the window function...
    wx = np.meshgrid(*([np.sin(k*l/2.)/(k*l/2.)] for k, l in zip(freq, boxlength))) ## array (dim, N, ..., N); each component ks[i] is the wavenumber map of i-th dimension
    window = np.ones(wx[0].shape)
    for w in wx:
        window = np.multiply(window, w)
    P = P / window**2
    """ 

    # compute angular power spectrum
    Pk, k, var = angular_average_nd(P, freq, nbins, log_bins=log_bins)

    return Pk, k, var

import numpy as np
import warnings

def cylindrical_average(field, coords, nbins, log_bins=False, use_same_bins=False):
    """
    3次元データ field (Nx, Ny, Nz) および各軸の座標 coords = [x, y, z] 
    に対し、横平面の距離 r = sqrt(x^2+y^2) と z = |z| による2次元ビニングを行い、
    各ビン内の平均と分散を求める関数。

    Parameters:
      field    : 3次元の実数型配列 (Nx,Ny,Nz)
      coords   : 各軸の座標情報のリスト [x, y, z] (それぞれ (Nx,), (Ny,), (Nz,))
      nbins    : 各軸方向のビン数（出力は (nbins, nbins) となる）
      log_bins : 横軸・縦軸ともに対数ビンにする場合は True (初期値 False)

    Returns:
      mean     : 各ビン内の平均値 (nbins, nbins)
      bins_r   : r 軸のビンエッジ配列 (nbins+1,)
      bins_z   : z 軸のビンエッジ配列 (nbins+1,)
      var      : 各ビン内の不偏分散 (nbins, nbins)
    """

    # 入力チェック：fieldが3次元、coordsが3個の1次元配列であること
    if field.ndim != 3 or len(coords) != 3:
        raise ValueError("field は3次元、coords は3つの軸情報のリストである必要があります。")

    # 各軸の座標グリッドを生成 (indexing='ij' を指定して元のshapeに合わせる)
    X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')

    # 新たな座標値の計算
    r_values = np.sqrt(X**2 + Y**2)
    z_values = np.abs(Z)

        
    # binning
    z_min, z_max = z_values.min(), z_values.max()
    r_min, r_max = r_values.min(), r_values.max()    
    if use_same_bins:
        z_min = min(z_min, r_min)
        z_max = max(z_max, r_max)
        r_min = z_min
        r_max = z_max
        
    if not log_bins:
        bins_z = np.linspace(z_min, z_max, nbins + 1)
    else:
        z_positive = z_values[z_values > 0]
        z_min_nonzero = z_positive.min()
        bins_z = np.logspace(np.log10(z_min_nonzero), np.log10(z_max), nbins + 1)
    
    if not log_bins:
        bins_r = np.linspace(r_min, r_max, nbins + 1)
    else:
        r_positive = r_values[r_values > 0]
        r_min_nonzero = r_positive.min()
        bins_r = np.logspace(np.log10(r_min_nonzero), np.log10(r_max), nbins + 1)

    # --- 各画素のビン番号の決定 ---
    # flatten して1次元配列として扱う
    r_flat = r_values.flatten()
    z_flat = z_values.flatten()
    field_flat = np.real(field.flatten())

    # digitize は1始まりのインデックスを返す（ビン範囲外の値は 0 または nbins+1）
    indx_r = np.digitize(r_flat, bins_r)
    indx_z = np.digitize(z_flat, bins_z)

    # 有効なビン内の値のみを抽出 (indx 1～nbinsに属するもの)
    valid = (indx_r >= 1) & (indx_r <= nbins) & (indx_z >= 1) & (indx_z <= nbins)
    # 0-based に変換: 各軸ともに -1 する
    valid_r = indx_r[valid] - 1  
    valid_z = indx_z[valid] - 1

    # 2次元のビン番号を一意にするために、合成インデックスを作成
    combined_idx = valid_r + nbins * valid_z

    # --- 各ビンに含まれる画素数の算出 ---
    counts = np.bincount(combined_idx, minlength=nbins * nbins)
    counts = counts.reshape((nbins, nbins))
    if np.any(counts == 0):
        warnings.warn("一部のビンに画素が存在しません。nbins を減らすか座標範囲を見直してください。")

    # --- 各ビンの field 値の総和を計算 ---
    sum_field = np.bincount(combined_idx, weights=field_flat[valid], minlength=nbins * nbins)
    sum_field = sum_field.reshape((nbins, nbins))

    # --- 各ビン内の平均値の計算 ---
    mean = np.empty((nbins, nbins))
    # 画素が存在するビンについては平均計算、無いところは NaN
    mean[counts > 0] = sum_field[counts > 0] / counts[counts > 0]
    mean[counts == 0] = np.nan

    # --- 分散の計算 ---
    # 各有効画素について、そのビンの平均値との差の二乗を計算
    # まず，flatten した2次元平均値（合成したときの順番で）から，各画素に対応する平均を lookup する
    average_arr = mean.flatten()  # shape (nbins*nbins,)
    pixel_means = average_arr[combined_idx]
    diff2 = (field_flat[valid] - pixel_means) ** 2
    sum_diff2 = np.bincount(combined_idx, weights=diff2, minlength=nbins * nbins)
    sum_diff2 = sum_diff2.reshape((nbins, nbins))
    var = np.empty((nbins, nbins))
    # 画素数が2以上のビンで不偏分散（ddof=1）を算出し、そうでなければ NaN
    var[counts > 1] = sum_diff2[counts > 1] / (counts[counts > 1] - 1)
    var[counts <= 1] = np.nan

    return mean, bins_r, bins_z, var


def compute_cylindrical_power(deltax, deltax2=None, boxlength=1., nbins=10, log_bins=False, use_same_bins=False):
    ## compmute 2d power spectrum (kpara, kperp)
    # Inputs:
    #   deltax: input image 
    #   deltax2: second input image. If this is not None, the cross-power spectrum will be computed
    #   boxlength: float or list of floats. The physical length of the side(s) in real space
    #   nbins: int. number of bins
    #   log_bins: use log-scale bins if True.
    #
    # Outputs: 
    #   Pk: angular-averaged power spectrum. 
    #   k: edge values of the bins. Note there are N+1 values when the length of Pk is N.
    #   var: variance 

    if deltax2 is not None and deltax.shape != deltax2.shape:
        raise ValueError("deltax and deltax2 must have the same shpae!")

    dim = len(deltax.shape)

    if not np.iterable(boxlength):
        boxlength = [boxlength] * dim
    V = np.prod(boxlength)

    # Fourier transformation
    FT, freq = my_fft(deltax, L=boxlength) 

    if deltax2 is not None:
        FT2, _ = my_fft(deltax2, L=boxlength)
    else:
        FT2 = FT

    # compute power spectrum
    P = np.real(FT * np.conj(FT2)) / V

    # compute angular power spectrum
    Pk, k_perp, k_para, var = cylindrical_average(P, freq, nbins, log_bins=log_bins, use_same_bins=use_same_bins)

    return Pk, k_perp, k_para, var

def normalized_power(deltax, deltax2=None, boxlength=1., nbins=20, log_bins=True):
    # compute normalized power spectrum
    # Pk / k^3
    #print(deltax.shape, boxlength)
    Pk, k, var = compute_power(deltax, deltax2=deltax2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    err = np.sqrt(var)

    if log_bins:
        k_values = 10**((np.log10(k[1:]) + np.log10(k[:-1])) * 0.5)
    else:
        k_values = (k[1:] + k[:-1]) * 0.5

    dim = len(deltax.shape)
    if dim == 1:
        norm_Pk = Pk * k_values
        norm_err = err * k_values
    elif dim == 2:
        norm_Pk = Pk * k_values**2 / (2*np.pi)
        norm_err = err * k_values**2 / (2*np.pi)
    elif dim == 3:
        norm_Pk = Pk * k_values**3 / (2*np.pi**2)
        norm_err = err * k_values**3 / (2*np.pi**2)
    else:
        print("Error: dim > 3 is not supported", file=sys.stderr)
        sys.exit(1)

    return norm_Pk, k, norm_err**2

def compute_r(image1, image2, boxlength=1., nbins=20, log_bins=True):
    # compute cross-correlation coefficient between image1 and image2: r = Px / sqrt( P1 * P2 )
    # boxlength: float or list of floats. The physical length of the side(s) in real space
    # nbins: int. number of bins
    # log_bins: use log-scale bins if True.
    Px, k, varx = compute_power(image1, deltax2=image2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    P1, _, var1 = compute_power(image1, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    P2, _, var2 = compute_power(image2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    return Px / np.sqrt( P1 * P2 ), k

