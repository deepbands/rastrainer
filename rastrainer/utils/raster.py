import os.path as osp
from typing import List, Tuple, Union, Optional
import numpy as np
from skimage import exposure

try:
    from osgeo import gdal
except:
    import gdal


class Raster:
    def __init__(self,
                 path: Union[str, gdal.Dataset],
                 band_list: Union[List[int], Tuple[int], None]=None) -> None:
        """ Class of read raster.
        Args:
            path (Union[str, gdal.Dataset]): The path of raster.
            band_list (Union[List[int], Tuple[int], None], optional): 
                band list (start with 1) or None (all of bands). Defaults to None.
        """
        super(Raster, self).__init__()
        if isinstance(path, str):
            if osp.exists(path):
                self.path = path
                self.ext_type = path.split(".")[-1]
                try:
                    # raster format support in GDAL: 
                    # https://www.osgeo.cn/gdal/drivers/raster/index.html
                    self._src_data = gdal.Open(path)
                except:
                    raise TypeError(
                        "Unsupported data format: `{}`".format(self.ext_type))
            else:
                raise ValueError("The path {0} not exists.".format(path))
        else:
            self._src_data = path
            self.path = None
            self.ext_type = None
        # init
        self._getInfo()
        self.setBands(band_list)
        self.datatype = Raster.get_type(self.getArray([0, 0], [1, 1]).dtype.name)

    def setBands(self, band_list: Union[List[int], Tuple[int], None]) -> None:
        """ Set band of data.
        Args:
            band_list (Union[List[int], Tuple[int], None]): 
                band list (start with 1) or None (all of bands).
        """
        if band_list is not None:
            if len(band_list) > self.bands:
                raise ValueError(
                    "The lenght of band_list must be less than {0}.".format(
                        str(self.bands)))
            if max(band_list) > self.bands or min(band_list) < 1:
                raise ValueError("The range of band_list must within [1, {0}].".
                                 format(str(self.bands)))
        self.band_list = band_list

    def getArray(
            self,
            start_loc: Union[List[int], Tuple[int, int], None]=None,
            block_size: Union[List[int], Tuple[int, int]]=[512, 512]) -> np.ndarray:
        """ Get ndarray data 
        Args:
            start_loc (Union[List[int], Tuple[int], None], optional): 
                Coordinates of the upper left corner of the block, if None means return full image.
            block_size (Union[List[int], Tuple[int]], optional): 
                Block size. Defaults to [512, 512].
        Returns:
            np.ndarray: data's ndarray.
        """
        if start_loc is None:
            return self._getArray()
        else:
            return self._getBlock(start_loc, block_size)

    def _getInfo(self) -> None:
        self.width = self._src_data.RasterXSize
        self.height = self._src_data.RasterYSize
        self.bands = self._src_data.RasterCount
        self.geot = self._src_data.GetGeoTransform()
        self.proj = self._src_data.GetProjection()

    def _getArray(
            self,
            window: Union[None, List[int], Tuple[int, int, int, int]]=None) -> np.ndarray:
        if self._src_data is None:
            raise ValueError("The raster is None.")
        if window is not None:
            xoff, yoff, xsize, ysize = window
        if self.band_list is None:
            if window is None:
                ima = self._src_data.ReadAsArray()
            else:
                ima = self._src_data.ReadAsArray(xoff, yoff, xsize, ysize)
        else:
            band_array = []
            for b in self.band_list:
                if window is None:
                    band_i = self._src_data.GetRasterBand(b).ReadAsArray()
                else:
                    band_i = self._src_data.GetRasterBand(b).ReadAsArray(
                        xoff, yoff, xsize, ysize)
                band_array.append(band_i)
            ima = np.stack(band_array, axis=0)
        if self.bands == 1:
            if len(ima.shape) == 3:
                ima = ima.squeeze(0)
        else:
            ima = ima.transpose((1, 2, 0))
        return Raster.to_uint8(ima)

    def _getBlock(
            self,
            start_loc: Union[List[int], Tuple[int, int]],
            block_size: Union[List[int], Tuple[int, int]]=[512, 512]) -> np.ndarray:
        if len(start_loc) != 2 or len(block_size) != 2:
            raise ValueError("The length start_loc/block_size must be 2.")
        xoff, yoff = start_loc
        xsize, ysize = block_size
        if (xoff < 0 or xoff > self.width) or (yoff < 0 or yoff > self.height):
            raise ValueError("start_loc must be within [0-{0}, 0-{1}].".format(
                str(self.width), str(self.height)))
        if xoff + xsize > self.width:
            xsize = self.width - xoff
        if yoff + ysize > self.height:
            ysize = self.height - yoff
        ima = self._getArray([int(xoff), int(yoff), int(xsize), int(ysize)])
        h, w = ima.shape[:2] if len(ima.shape) == 3 else ima.shape
        if self.bands != 1:
            tmp = np.zeros(
                (block_size[0], block_size[1], self.bands), dtype=ima.dtype)
            tmp[:h, :w, :] = ima
        else:
            tmp = np.zeros((block_size[0], block_size[1]), dtype=ima.dtype)
            tmp[:h, :w] = ima
        return tmp

    @classmethod
    def get_type(cls, type_name: str) -> int:
        if type_name in ["bool", "uint8"]:
            gdal_type = gdal.GDT_Byte
        elif type_name in ["int8", "int16"]:
            gdal_type = gdal.GDT_Int16
        elif type_name == "uint16":
            gdal_type = gdal.GDT_UInt16
        elif type_name == "int32":
            gdal_type = gdal.GDT_Int32
        elif type_name == "uint32":
            gdal_type = gdal.GDT_UInt32
        elif type_name in ["int64", "uint64", "float16", "float32"]:
            gdal_type = gdal.GDT_Float32
        elif type_name == "float64":
            gdal_type = gdal.GDT_Float64
        elif type_name == "complex64":
            gdal_type = gdal.GDT_CFloat64
        else:
            raise TypeError("Non-suported data type `{}`.".format(type_name))
        return gdal_type

    @classmethod
    def to_uint8(cls, im, is_linear=False):
        # 2% linear stretch
        def _two_percent_linear(image, max_out=255, min_out=0):
            def _gray_process(gray, maxout=max_out, minout=min_out):
                # get the corresponding gray level at 98% histogram
                high_value = np.percentile(gray, 98)
                low_value = np.percentile(gray, 2)
                truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
                processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * \
                                (maxout - minout)
                return np.uint8(processed_gray)

            if len(image.shape) == 3:
                processes = []
                for b in range(image.shape[-1]):
                    processes.append(_gray_process(image[:, :, b]))
                result = np.stack(processes, axis=2)
            else:  # if len(image.shape) == 2
                result = _gray_process(image)
            return np.uint8(result)

        # simple image standardization
        def _sample_norm(image):
            stretches = []
            if len(image.shape) == 3:
                for b in range(image.shape[-1]):
                    stretched = exposure.equalize_hist(image[:, :, b])
                    stretched /= float(np.max(stretched))
                    stretches.append(stretched)
                stretched_img = np.stack(stretches, axis=2)
            else:  # if len(image.shape) == 2
                stretched_img = exposure.equalize_hist(image)
            return np.uint8(stretched_img * 255)

        dtype = im.dtype.name
        if dtype != "uint8":
            im = _sample_norm(im)
        if is_linear:
            im = _two_percent_linear(im)
        return im

    @classmethod
    def save_geotiff(cls,
                     image: np.ndarray, 
                     save_path: str, 
                     proj: str, 
                     geotf: Tuple,
                     use_type: Optional[int]=None,
                     clear_ds: bool=True) -> None:
        if len(image.shape) == 2:
            height, width = image.shape
            channel = 1
        else:
            height, width, channel = image.shape
        if use_type is not None:
            data_type = use_type
        else:
            data_type = Raster.get_type(image.dtype.name)
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(save_path, width, height, channel, data_type)
        dst_ds.SetGeoTransform(geotf)
        dst_ds.SetProjection(proj)
        if channel > 1:
            for i in range(channel):
                band = dst_ds.GetRasterBand(i + 1)
                band.WriteArray(image[:, :, i])
                dst_ds.FlushCache()
        else:
            band = dst_ds.GetRasterBand(1)
            band.WriteArray(image)
            dst_ds.FlushCache()
        if clear_ds:
            dst_ds = None 
