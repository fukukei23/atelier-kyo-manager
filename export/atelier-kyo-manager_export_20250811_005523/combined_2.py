
# === atelier-kyo-manager/app\utils\ai_generate_descriptions.py ===
import os
import requests
import csv
import time
import random
from dotenv import load_dotenv
from datetime import datetime

# 環境変数の読み込み
load_dotenv()

class BUYMADescriptionGenerator:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.endpoint = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BUYMA-Auto-Desc/3.0"
        }
        self.model = "sonar-pro"  # 最新有効モデル名

    def _create_log(self, message, log_type="error"):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_type}_{timestamp}.log"
        with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
            f.write(message)

    def _call_api(self, payload):
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code}\n{e.response.text}"
            self._create_log(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Failed: {str(e)}"
            self._create_log(error_msg)
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            self._create_log(error_msg)
        return None

    def generate_description(self, product_info):
        prompt = f"""
        あなたはBUYMA専門のプロコピーライターです。
        以下の条件で商品説明文を日本語で作成してください。

        [要件]
        1. 300-500文字
        2. ブランド名とカテゴリを自然に配置
        3. 商品の特徴を3点箇条書き
        4. 感情に訴える表現（例: 「上質なレザーの風合い」）
        5. 急かし表現や「今すぐ」などの強い勧誘は避ける
        6. 重複表現を避け、情報を簡潔に整理
        7. 生活シーンや使用イメージを盛り込む
        8. 読み手に自然に提案する口調で仕上げる

        [商品情報]
        ブランド: {product_info.get('brand', '')}
        商品名: {product_info.get('name', '')}
        カテゴリ: {product_info.get('category', '')}
        特徴: {product_info.get('features', '')}
        素材: {product_info.get('materials', '')}
        サイズ: {product_info.get('size', '')}
        カラー: {product_info.get('color', '')}
        対象層: {product_info.get('target', '')}
        価格帯: {product_info.get('price_range', '')}
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "あなたは高級ファッション専門のコピーライターです。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        result = self._call_api(payload)
        return result['choices'][0]['message']['content'].strip() if result else None

    def batch_process(self, products, retry=3):
        results = []
        for idx, product in enumerate(products):
            attempt = 0
            while attempt < retry:
                try:
                    print(f"処理中 ({idx+1}/{len(products)})")
                    desc = self.generate_description(product)
                    if desc:
                        results.append({**product, "description": desc})
                        break
                    else:
                        attempt += 1
                        time.sleep(2 ** attempt)
                except Exception as e:
                    print(f"エラー: {str(e)}")
                    attempt += 1
            time.sleep(random.uniform(1, 3))
        return results

if __name__ == "__main__":
    generator = BUYMADescriptionGenerator()
    sample_products = [
        {
            "brand": "PRADA",
            "name": "シンフォニー トートバッグ",
            "category": "トートバッグ",
            "features": "軽量ナイロン素材、大容量収納",
            "materials": "ナイロン",
            "size": "W35cm x H25cm x D15cm",
            "color": "ブラック",
            "target": "20-30代女性",
            "price_range": "15-20万円"
        }
    ]
    processed_data = generator.batch_process(sample_products)
    if processed_data:
        output_path = os.path.join(os.path.dirname(__file__), 'output.csv')
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=processed_data[0].keys())
            writer.writeheader()
            writer.writerows(processed_data)
        print(f"CSVファイルを正常に出力しました: {output_path}")
    else:
        print("処理に失敗しました。ログを確認してください。")

# === atelier-kyo-manager/app\utils\utils\ai_generate_descriptions.py ===
import os
import requests
import csv
import time
import random
from dotenv import load_dotenv
from datetime import datetime

# 環境変数の読み込み
load_dotenv()

class BUYMADescriptionGenerator:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.endpoint = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BUYMA-Auto-Desc/3.0"
        }
        self.model = "sonar-pro"  # 最新有効モデル名

    def _create_log(self, message, log_type="error"):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_type}_{timestamp}.log"
        with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
            f.write(message)

    def _call_api(self, payload):
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code}\n{e.response.text}"
            self._create_log(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Failed: {str(e)}"
            self._create_log(error_msg)
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            self._create_log(error_msg)
        return None

    def generate_description(self, product_info):
        prompt = f"""
        あなたはBUYMA専門のプロコピーライターです。
        以下の条件で商品説明文を日本語で作成してください。

        [要件]
        1. 300-500文字
        2. ブランド名とカテゴリを自然に配置
        3. 商品の特徴を3点箇条書き
        4. 感情に訴える表現（例: 「上質なレザーの風合い」）
        5. 急かし表現や「今すぐ」などの強い勧誘は避ける
        6. 重複表現を避け、情報を簡潔に整理
        7. 生活シーンや使用イメージを盛り込む
        8. 読み手に自然に提案する口調で仕上げる

        [商品情報]
        ブランド: {product_info.get('brand', '')}
        商品名: {product_info.get('name', '')}
        カテゴリ: {product_info.get('category', '')}
        特徴: {product_info.get('features', '')}
        素材: {product_info.get('materials', '')}
        サイズ: {product_info.get('size', '')}
        カラー: {product_info.get('color', '')}
        対象層: {product_info.get('target', '')}
        価格帯: {product_info.get('price_range', '')}
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "あなたは高級ファッション専門のコピーライターです。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        result = self._call_api(payload)
        return result['choices'][0]['message']['content'].strip() if result else None

    def batch_process(self, products, retry=3):
        results = []
        for idx, product in enumerate(products):
            attempt = 0
            while attempt < retry:
                try:
                    print(f"処理中 ({idx+1}/{len(products)})")
                    desc = self.generate_description(product)
                    if desc:
                        results.append({**product, "description": desc})
                        break
                    else:
                        attempt += 1
                        time.sleep(2 ** attempt)
                except Exception as e:
                    print(f"エラー: {str(e)}")
                    attempt += 1
            time.sleep(random.uniform(1, 3))
        return results

if __name__ == "__main__":
    generator = BUYMADescriptionGenerator()
    sample_products = [
        {
            "brand": "PRADA",
            "name": "シンフォニー トートバッグ",
            "category": "トートバッグ",
            "features": "軽量ナイロン素材、大容量収納",
            "materials": "ナイロン",
            "size": "W35cm x H25cm x D15cm",
            "color": "ブラック",
            "target": "20-30代女性",
            "price_range": "15-20万円"
        }
    ]
    processed_data = generator.batch_process(sample_products)
    if processed_data:
        output_path = os.path.join(os.path.dirname(__file__), 'output.csv')
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=processed_data[0].keys())
            writer.writeheader()
            writer.writerows(processed_data)
        print(f"CSVファイルを正常に出力しました: {output_path}")
    else:
        print("処理に失敗しました。ログを確認してください。")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_io.py ===
import gc
import gzip
import locale
import os
import re
import sys
import threading
import time
import warnings
from ctypes import c_bool
from datetime import datetime
from io import BytesIO, StringIO
from multiprocessing import Value, get_context
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

import numpy as np
import numpy.ma as ma
from numpy._utils import asbytes
from numpy.exceptions import VisibleDeprecationWarning
from numpy.lib import _npyio_impl
from numpy.lib._iotools import ConversionWarning, ConverterError
from numpy.lib._npyio_impl import recfromcsv, recfromtxt
from numpy.ma.testutils import assert_equal
from numpy.testing import (
    HAS_REFCOUNT,
    IS_PYPY,
    IS_WASM,
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_no_gc_cycles,
    assert_no_warnings,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    break_cycles,
    suppress_warnings,
    tempdir,
    temppath,
)
from numpy.testing._private.utils import requires_memory


class TextIO(BytesIO):
    """Helper IO class.

    Writes encode strings to bytes if needed, reads return bytes.
    This makes it easier to emulate files opened in binary mode
    without needing to explicitly convert strings to bytes in
    setting up the test data.

    """
    def __init__(self, s=""):
        BytesIO.__init__(self, asbytes(s))

    def write(self, s):
        BytesIO.write(self, asbytes(s))

    def writelines(self, lines):
        BytesIO.writelines(self, [asbytes(s) for s in lines])


IS_64BIT = sys.maxsize > 2**32
try:
    import bz2
    HAS_BZ2 = True
except ImportError:
    HAS_BZ2 = False
try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False


def strptime(s, fmt=None):
    """
    This function is available in the datetime module only from Python >=
    2.5.

    """
    if isinstance(s, bytes):
        s = s.decode("latin1")
    return datetime(*time.strptime(s, fmt)[:3])


class RoundtripTest:
    def roundtrip(self, save_func, *args, **kwargs):
        """
        save_func : callable
            Function used to save arrays to file.
        file_on_disk : bool
            If true, store the file on disk, instead of in a
            string buffer.
        save_kwds : dict
            Parameters passed to `save_func`.
        load_kwds : dict
            Parameters passed to `numpy.load`.
        args : tuple of arrays
            Arrays stored to file.

        """
        save_kwds = kwargs.get('save_kwds', {})
        load_kwds = kwargs.get('load_kwds', {"allow_pickle": True})
        file_on_disk = kwargs.get('file_on_disk', False)

        if file_on_disk:
            target_file = NamedTemporaryFile(delete=False)
            load_file = target_file.name
        else:
            target_file = BytesIO()
            load_file = target_file

        try:
            arr = args

            save_func(target_file, *arr, **save_kwds)
            target_file.flush()
            target_file.seek(0)

            if sys.platform == 'win32' and not isinstance(target_file, BytesIO):
                target_file.close()

            arr_reloaded = np.load(load_file, **load_kwds)

            self.arr = arr
            self.arr_reloaded = arr_reloaded
        finally:
            if not isinstance(target_file, BytesIO):
                target_file.close()
                # holds an open file descriptor so it can't be deleted on win
                if 'arr_reloaded' in locals():
                    if not isinstance(arr_reloaded, np.lib.npyio.NpzFile):
                        os.remove(target_file.name)

    def check_roundtrips(self, a):
        self.roundtrip(a)
        self.roundtrip(a, file_on_disk=True)
        self.roundtrip(np.asfortranarray(a))
        self.roundtrip(np.asfortranarray(a), file_on_disk=True)
        if a.shape[0] > 1:
            # neither C nor Fortran contiguous for 2D arrays or more
            self.roundtrip(np.asfortranarray(a)[1:])
            self.roundtrip(np.asfortranarray(a)[1:], file_on_disk=True)

    def test_array(self):
        a = np.array([], float)
        self.check_roundtrips(a)

        a = np.array([[1, 2], [3, 4]], float)
        self.check_roundtrips(a)

        a = np.array([[1, 2], [3, 4]], int)
        self.check_roundtrips(a)

        a = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        self.check_roundtrips(a)

        a = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        self.check_roundtrips(a)

    def test_array_object(self):
        a = np.array([], object)
        self.check_roundtrips(a)

        a = np.array([[1, 2], [3, 4]], object)
        self.check_roundtrips(a)

    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        self.roundtrip(a)

    @pytest.mark.skipif(sys.platform == 'win32', reason="Fails on Win32")
    def test_mmap(self):
        a = np.array([[1, 2.5], [4, 7.3]])
        self.roundtrip(a, file_on_disk=True, load_kwds={'mmap_mode': 'r'})

        a = np.asfortranarray([[1, 2.5], [4, 7.3]])
        self.roundtrip(a, file_on_disk=True, load_kwds={'mmap_mode': 'r'})

    def test_record(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        self.check_roundtrips(a)

    @pytest.mark.slow
    def test_format_2_0(self):
        dt = [(("%d" % i) * 100, float) for i in range(500)]
        a = np.ones(1000, dtype=dt)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always', '', UserWarning)
            self.check_roundtrips(a)


class TestSaveLoad(RoundtripTest):
    def roundtrip(self, *args, **kwargs):
        RoundtripTest.roundtrip(self, np.save, *args, **kwargs)
        assert_equal(self.arr[0], self.arr_reloaded)
        assert_equal(self.arr[0].dtype, self.arr_reloaded.dtype)
        assert_equal(self.arr[0].flags.fnc, self.arr_reloaded.flags.fnc)


class TestSavezLoad(RoundtripTest):
    def roundtrip(self, *args, **kwargs):
        RoundtripTest.roundtrip(self, np.savez, *args, **kwargs)
        try:
            for n, arr in enumerate(self.arr):
                reloaded = self.arr_reloaded['arr_%d' % n]
                assert_equal(arr, reloaded)
                assert_equal(arr.dtype, reloaded.dtype)
                assert_equal(arr.flags.fnc, reloaded.flags.fnc)
        finally:
            # delete tempfile, must be done here on windows
            if self.arr_reloaded.fid:
                self.arr_reloaded.fid.close()
                os.remove(self.arr_reloaded.fid.name)

    @pytest.mark.skipif(IS_PYPY, reason="Hangs on PyPy")
    @pytest.mark.skipif(not IS_64BIT, reason="Needs 64bit platform")
    @pytest.mark.slow
    def test_big_arrays(self):
        L = (1 << 31) + 100000
        a = np.empty(L, dtype=np.uint8)
        with temppath(prefix="numpy_test_big_arrays_", suffix=".npz") as tmp:
            np.savez(tmp, a=a)
            del a
            npfile = np.load(tmp)
            a = npfile['a']  # Should succeed
            npfile.close()

    def test_multiple_arrays(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        self.roundtrip(a, b)

    def test_named_arrays(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        c = BytesIO()
        np.savez(c, file_a=a, file_b=b)
        c.seek(0)
        l = np.load(c)
        assert_equal(a, l['file_a'])
        assert_equal(b, l['file_b'])

    def test_tuple_getitem_raises(self):
        # gh-23748
        a = np.array([1, 2, 3])
        f = BytesIO()
        np.savez(f, a=a)
        f.seek(0)
        l = np.load(f)
        with pytest.raises(KeyError, match="(1, 2)"):
            l[1, 2]

    def test_BagObj(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        c = BytesIO()
        np.savez(c, file_a=a, file_b=b)
        c.seek(0)
        l = np.load(c)
        assert_equal(sorted(dir(l.f)), ['file_a', 'file_b'])
        assert_equal(a, l.f.file_a)
        assert_equal(b, l.f.file_b)

    @pytest.mark.skipif(IS_WASM, reason="Cannot start thread")
    def test_savez_filename_clashes(self):
        # Test that issue #852 is fixed
        # and savez functions in multithreaded environment

        def writer(error_list):
            with temppath(suffix='.npz') as tmp:
                arr = np.random.randn(500, 500)
                try:
                    np.savez(tmp, arr=arr)
                except OSError as err:
                    error_list.append(err)

        errors = []
        threads = [threading.Thread(target=writer, args=(errors,))
                   for j in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise AssertionError(errors)

    def test_not_closing_opened_fid(self):
        # Test that issue #2178 is fixed:
        # verify could seek on 'loaded' file
        with temppath(suffix='.npz') as tmp:
            with open(tmp, 'wb') as fp:
                np.savez(fp, data='LOVELY LOAD')
            with open(tmp, 'rb', 10000) as fp:
                fp.seek(0)
                assert_(not fp.closed)
                np.load(fp)['data']
                # fp must not get closed by .load
                assert_(not fp.closed)
                fp.seek(0)
                assert_(not fp.closed)

    @pytest.mark.slow_pypy
    def test_closing_fid(self):
        # Test that issue #1517 (too many opened files) remains closed
        # It might be a "weak" test since failed to get triggered on
        # e.g. Debian sid of 2012 Jul 05 but was reported to
        # trigger the failure on Ubuntu 10.04:
        # http://projects.scipy.org/numpy/ticket/1517#comment:2
        with temppath(suffix='.npz') as tmp:
            np.savez(tmp, data='LOVELY LOAD')
            # We need to check if the garbage collector can properly close
            # numpy npz file returned by np.load when their reference count
            # goes to zero.  Python running in debug mode raises a
            # ResourceWarning when file closing is left to the garbage
            # collector, so we catch the warnings.
            with suppress_warnings() as sup:
                sup.filter(ResourceWarning)  # TODO: specify exact message
                for i in range(1, 1025):
                    try:
                        np.load(tmp)["data"]
                    except Exception as e:
                        msg = f"Failed to load data from a file: {e}"
                        raise AssertionError(msg)
                    finally:
                        if IS_PYPY:
                            gc.collect()

    def test_closing_zipfile_after_load(self):
        # Check that zipfile owns file and can close it.  This needs to
        # pass a file name to load for the test. On windows failure will
        # cause a second error will be raised when the attempt to remove
        # the open file is made.
        prefix = 'numpy_test_closing_zipfile_after_load_'
        with temppath(suffix='.npz', prefix=prefix) as tmp:
            np.savez(tmp, lab='place holder')
            data = np.load(tmp)
            fp = data.zip.fp
            data.close()
            assert_(fp.closed)

    @pytest.mark.parametrize("count, expected_repr", [
        (1, "NpzFile {fname!r} with keys: arr_0"),
        (5, "NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4"),
        # _MAX_REPR_ARRAY_COUNT is 5, so files with more than 5 keys are
        # expected to end in '...'
        (6, "NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4..."),
    ])
    def test_repr_lists_keys(self, count, expected_repr):
        a = np.array([[1, 2], [3, 4]], float)
        with temppath(suffix='.npz') as tmp:
            np.savez(tmp, *[a] * count)
            l = np.load(tmp)
            assert repr(l) == expected_repr.format(fname=tmp)
            l.close()


class TestSaveTxt:
    def test_array(self):
        a = np.array([[1, 2], [3, 4]], float)
        fmt = "%.18e"
        c = BytesIO()
        np.savetxt(c, a, fmt=fmt)
        c.seek(0)
        assert_equal(c.readlines(),
                     [asbytes((fmt + ' ' + fmt + '\n') % (1, 2)),
                      asbytes((fmt + ' ' + fmt + '\n') % (3, 4))])

        a = np.array([[1, 2], [3, 4]], int)
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1 2\n', b'3 4\n'])

    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'1\n', b'2\n', b'3\n', b'4\n'])

    def test_0D_3D(self):
        c = BytesIO()
        assert_raises(ValueError, np.savetxt, c, np.array(1))
        assert_raises(ValueError, np.savetxt, c, np.array([[[1], [2]]]))

    def test_structured(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1 2\n', b'3 4\n'])

    def test_structured_padded(self):
        # gh-13297
        a = np.array([(1, 2, 3), (4, 5, 6)], dtype=[
            ('foo', 'i4'), ('bar', 'i4'), ('baz', 'i4')
        ])
        c = BytesIO()
        np.savetxt(c, a[['foo', 'baz']], fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1 3\n', b'4 6\n'])

    def test_multifield_view(self):
        a = np.ones(1, dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
        v = a[['x', 'z']]
        with temppath(suffix='.npy') as path:
            path = Path(path)
            np.save(path, v)
            data = np.load(path)
            assert_array_equal(data, v)

    def test_delimiter(self):
        a = np.array([[1., 2.], [3., 4.]])
        c = BytesIO()
        np.savetxt(c, a, delimiter=',', fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1,2\n', b'3,4\n'])

    def test_format(self):
        a = np.array([(1, 2), (3, 4)])
        c = BytesIO()
        # Sequence of formats
        np.savetxt(c, a, fmt=['%02d', '%3.1f'])
        c.seek(0)
        assert_equal(c.readlines(), [b'01 2.0\n', b'03 4.0\n'])

        # A single multiformat string
        c = BytesIO()
        np.savetxt(c, a, fmt='%02d : %3.1f')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'01 : 2.0\n', b'03 : 4.0\n'])

        # Specify delimiter, should be overridden
        c = BytesIO()
        np.savetxt(c, a, fmt='%02d : %3.1f', delimiter=',')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'01 : 2.0\n', b'03 : 4.0\n'])

        # Bad fmt, should raise a ValueError
        c = BytesIO()
        assert_raises(ValueError, np.savetxt, c, a, fmt=99)

    def test_header_footer(self):
        # Test the functionality of the header and footer keyword argument.

        c = BytesIO()
        a = np.array([(1, 2), (3, 4)], dtype=int)
        test_header_footer = 'Test header / footer'
        # Test the header keyword argument
        np.savetxt(c, a, fmt='%1d', header=test_header_footer)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes('# ' + test_header_footer + '\n1 2\n3 4\n'))
        # Test the footer keyword argument
        c = BytesIO()
        np.savetxt(c, a, fmt='%1d', footer=test_header_footer)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes('1 2\n3 4\n# ' + test_header_footer + '\n'))
        # Test the commentstr keyword argument used on the header
        c = BytesIO()
        commentstr = '% '
        np.savetxt(c, a, fmt='%1d',
                   header=test_header_footer, comments=commentstr)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes(commentstr + test_header_footer + '\n' + '1 2\n3 4\n'))
        # Test the commentstr keyword argument used on the footer
        c = BytesIO()
        commentstr = '% '
        np.savetxt(c, a, fmt='%1d',
                   footer=test_header_footer, comments=commentstr)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes('1 2\n3 4\n' + commentstr + test_header_footer + '\n'))

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_file_roundtrip(self, filename_type):
        with temppath() as name:
            a = np.array([(1, 2), (3, 4)])
            np.savetxt(filename_type(name), a)
            b = np.loadtxt(filename_type(name))
            assert_array_equal(a, b)

    def test_complex_arrays(self):
        ncols = 2
        nrows = 2
        a = np.zeros((ncols, nrows), dtype=np.complex128)
        re = np.pi
        im = np.e
        a[:] = re + 1.0j * im

        # One format only
        c = BytesIO()
        np.savetxt(c, a, fmt=' %+.3e')
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n',
             b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n'])

        # One format for each real and imaginary part
        c = BytesIO()
        np.savetxt(c, a, fmt='  %+.3e' * 2 * ncols)
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n',
             b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n'])

        # One format for each complex number
        c = BytesIO()
        np.savetxt(c, a, fmt=['(%.3e%+.3ej)'] * ncols)
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n',
             b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n'])

    def test_complex_negative_exponent(self):
        # Previous to 1.15, some formats generated x+-yj, gh 7895
        ncols = 2
        nrows = 2
        a = np.zeros((ncols, nrows), dtype=np.complex128)
        re = np.pi
        im = np.e
        a[:] = re - 1.0j * im
        c = BytesIO()
        np.savetxt(c, a, fmt='%.3e')
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b' (3.142e+00-2.718e+00j)  (3.142e+00-2.718e+00j)\n',
             b' (3.142e+00-2.718e+00j)  (3.142e+00-2.718e+00j)\n'])

    def test_custom_writer(self):

        class CustomWriter(list):
            def write(self, text):
                self.extend(text.split(b'\n'))

        w = CustomWriter()
        a = np.array([(1, 2), (3, 4)])
        np.savetxt(w, a)
        b = np.loadtxt(w)
        assert_array_equal(a, b)

    def test_unicode(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        with tempdir() as tmpdir:
            # set encoding as on windows it may not be unicode even on py3
            np.savetxt(os.path.join(tmpdir, 'test.csv'), a, fmt=['%s'],
                       encoding='UTF-8')

    def test_unicode_roundtrip(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        # our gz wrapper support encoding
        suffixes = ['', '.gz']
        if HAS_BZ2:
            suffixes.append('.bz2')
        if HAS_LZMA:
            suffixes.extend(['.xz', '.lzma'])
        with tempdir() as tmpdir:
            for suffix in suffixes:
                np.savetxt(os.path.join(tmpdir, 'test.csv' + suffix), a,
                           fmt=['%s'], encoding='UTF-16-LE')
                b = np.loadtxt(os.path.join(tmpdir, 'test.csv' + suffix),
                               encoding='UTF-16-LE', dtype=np.str_)
                assert_array_equal(a, b)

    def test_unicode_bytestream(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        s = BytesIO()
        np.savetxt(s, a, fmt=['%s'], encoding='UTF-8')
        s.seek(0)
        assert_equal(s.read().decode('UTF-8'), utf8 + '\n')

    def test_unicode_stringstream(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        s = StringIO()
        np.savetxt(s, a, fmt=['%s'], encoding='UTF-8')
        s.seek(0)
        assert_equal(s.read(), utf8 + '\n')

    @pytest.mark.parametrize("iotype", [StringIO, BytesIO])
    def test_unicode_and_bytes_fmt(self, iotype):
        # string type of fmt should not matter, see also gh-4053
        a = np.array([1.])
        s = iotype()
        np.savetxt(s, a, fmt="%f")
        s.seek(0)
        if iotype is StringIO:
            assert_equal(s.read(), "%f\n" % 1.)
        else:
            assert_equal(s.read(), b"%f\n" % 1.)

    @pytest.mark.skipif(sys.platform == 'win32', reason="files>4GB may not work")
    @pytest.mark.slow
    @requires_memory(free_bytes=7e9)
    def test_large_zip(self):
        def check_large_zip(memoryerror_raised):
            memoryerror_raised.value = False
            try:
                # The test takes at least 6GB of memory, writes a file larger
                # than 4GB. This tests the ``allowZip64`` kwarg to ``zipfile``
                test_data = np.asarray([np.random.rand(
                                        np.random.randint(50, 100), 4)
                                        for i in range(800000)], dtype=object)
                with tempdir() as tmpdir:
                    np.savez(os.path.join(tmpdir, 'test.npz'),
                             test_data=test_data)
            except MemoryError:
                memoryerror_raised.value = True
                raise
        # run in a subprocess to ensure memory is released on PyPy, see gh-15775
        # Use an object in shared memory to re-raise the MemoryError exception
        # in our process if needed, see gh-16889
        memoryerror_raised = Value(c_bool)

        # Since Python 3.8, the default start method for multiprocessing has
        # been changed from 'fork' to 'spawn' on macOS, causing inconsistency
        # on memory sharing model, leading to failed test for check_large_zip
        ctx = get_context('fork')
        p = ctx.Process(target=check_large_zip, args=(memoryerror_raised,))
        p.start()
        p.join()
        if memoryerror_raised.value:
            raise MemoryError("Child process raised a MemoryError exception")
        # -9 indicates a SIGKILL, probably an OOM.
        if p.exitcode == -9:
            pytest.xfail("subprocess got a SIGKILL, apparently free memory was not sufficient")
        assert p.exitcode == 0

class LoadTxtBase:
    def check_compressed(self, fopen, suffixes):
        # Test that we can load data from a compressed file
        wanted = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')
        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            for suffix in suffixes:
                with temppath(suffix=suffix) as name:
                    with fopen(name, mode='wt', encoding='UTF-32-LE') as f:
                        f.write(data)
                    res = self.loadfunc(name, encoding='UTF-32-LE')
                    assert_array_equal(res, wanted)
                    with fopen(name, "rt",  encoding='UTF-32-LE') as f:
                        res = self.loadfunc(f)
                    assert_array_equal(res, wanted)

    def test_compressed_gzip(self):
        self.check_compressed(gzip.open, ('.gz',))

    @pytest.mark.skipif(not HAS_BZ2, reason="Needs bz2")
    def test_compressed_bz2(self):
        self.check_compressed(bz2.open, ('.bz2',))

    @pytest.mark.skipif(not HAS_LZMA, reason="Needs lzma")
    def test_compressed_lzma(self):
        self.check_compressed(lzma.open, ('.xz', '.lzma'))

    def test_encoding(self):
        with temppath() as path:
            with open(path, "wb") as f:
                f.write('0.\n1.\n2.'.encode("UTF-16"))
            x = self.loadfunc(path, encoding="UTF-16")
            assert_array_equal(x, [0., 1., 2.])

    def test_stringload(self):
        # umlaute
        nonascii = b'\xc3\xb6\xc3\xbc\xc3\xb6'.decode("UTF-8")
        with temppath() as path:
            with open(path, "wb") as f:
                f.write(nonascii.encode("UTF-16"))
            x = self.loadfunc(path, encoding="UTF-16", dtype=np.str_)
            assert_array_equal(x, nonascii)

    def test_binary_decode(self):
        utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
        v = self.loadfunc(BytesIO(utf16), dtype=np.str_, encoding='UTF-16')
        assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))

    def test_converters_decode(self):
        # test converters that decode strings
        c = TextIO()
        c.write(b'\xcf\x96')
        c.seek(0)
        x = self.loadfunc(c, dtype=np.str_, encoding="bytes",
                          converters={0: lambda x: x.decode('UTF-8')})
        a = np.array([b'\xcf\x96'.decode('UTF-8')])
        assert_array_equal(x, a)

    def test_converters_nodecode(self):
        # test native string converters enabled by setting an encoding
        utf8 = b'\xcf\x96'.decode('UTF-8')
        with temppath() as path:
            with open(path, 'wt', encoding='UTF-8') as f:
                f.write(utf8)
            x = self.loadfunc(path, dtype=np.str_,
                              converters={0: lambda x: x + 't'},
                              encoding='UTF-8')
            a = np.array([utf8 + 't'])
            assert_array_equal(x, a)


class TestLoadTxt(LoadTxtBase):
    loadfunc = staticmethod(np.loadtxt)

    def setup_method(self):
        # lower chunksize for testing
        self.orig_chunk = _npyio_impl._loadtxt_chunksize
        _npyio_impl._loadtxt_chunksize = 1

    def teardown_method(self):
        _npyio_impl._loadtxt_chunksize = self.orig_chunk

    def test_record(self):
        c = TextIO()
        c.write('1 2\n3 4')
        c.seek(0)
        x = np.loadtxt(c, dtype=[('x', np.int32), ('y', np.int32)])
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_array_equal(x, a)

        d = TextIO()
        d.write('M 64 75.0\nF 25 60.0')
        d.seek(0)
        mydescriptor = {'names': ('gender', 'age', 'weight'),
                        'formats': ('S1', 'i4', 'f4')}
        b = np.array([('M', 64.0, 75.0),
                      ('F', 25.0, 60.0)], dtype=mydescriptor)
        y = np.loadtxt(d, dtype=mydescriptor)
        assert_array_equal(y, b)

    def test_array(self):
        c = TextIO()
        c.write('1 2\n3 4')

        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([[1, 2], [3, 4]], int)
        assert_array_equal(x, a)

        c.seek(0)
        x = np.loadtxt(c, dtype=float)
        a = np.array([[1, 2], [3, 4]], float)
        assert_array_equal(x, a)

    def test_1D(self):
        c = TextIO()
        c.write('1\n2\n3\n4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

        c = TextIO()
        c.write('1,2,3,4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

    def test_missing(self):
        c = TextIO()
        c.write('1,2,3,,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       converters={3: lambda s: int(s or - 999)})
        a = np.array([1, 2, 3, -999, 5], int)
        assert_array_equal(x, a)

    def test_converters_with_usecols(self):
        c = TextIO()
        c.write('1,2,3,,5\n6,7,8,9,10\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       converters={3: lambda s: int(s or - 999)},
                       usecols=(1, 3,))
        a = np.array([[2, -999], [7, 9]], int)
        assert_array_equal(x, a)

    def test_comments_unicode(self):
        c = TextIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments='#')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_comments_byte(self):
        c = TextIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments=b'#')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_comments_multiple(self):
        c = TextIO()
        c.write('# comment\n1,2,3\n@ comment2\n4,5,6 // comment3')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments=['#', '@', '//'])
        a = np.array([[1, 2, 3], [4, 5, 6]], int)
        assert_array_equal(x, a)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    def test_comments_multi_chars(self):
        c = TextIO()
        c.write('/* comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments='/*')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

        # Check that '/*' is not transformed to ['/', '*']
        c = TextIO()
        c.write('*/ comment\n1,2,3,5\n')
        c.seek(0)
        assert_raises(ValueError, np.loadtxt, c, dtype=int, delimiter=',',
                      comments='/*')

    def test_skiprows(self):
        c = TextIO()
        c.write('comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

        c = TextIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_usecols(self):
        a = np.array([[1, 2], [3, 4]], float)
        c = BytesIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1,))
        assert_array_equal(x, a[:, 1])

        a = np.array([[1, 2, 3], [3, 4, 5]], float)
        c = BytesIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1, 2))
        assert_array_equal(x, a[:, 1:])

        # Testing with arrays instead of tuples.
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=np.array([1, 2]))
        assert_array_equal(x, a[:, 1:])

        # Testing with an integer instead of a sequence
        for int_type in [int, np.int8, np.int16,
                         np.int32, np.int64, np.uint8, np.uint16,
                         np.uint32, np.uint64]:
            to_read = int_type(1)
            c.seek(0)
            x = np.loadtxt(c, dtype=float, usecols=to_read)
            assert_array_equal(x, a[:, 1])

        # Testing with some crazy custom integer type
        class CrazyInt:
            def __index__(self):
                return 1

        crazy_int = CrazyInt()
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=crazy_int)
        assert_array_equal(x, a[:, 1])

        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(crazy_int,))
        assert_array_equal(x, a[:, 1])

        # Checking with dtypes defined converters.
        data = '''JOE 70.1 25.3
                BOB 60.5 27.9
                '''
        c = TextIO(data)
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        arr = np.loadtxt(c, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        assert_equal(arr['stid'], [b"JOE", b"BOB"])
        assert_equal(arr['temp'], [25.3, 27.9])

        # Testing non-ints in usecols
        c.seek(0)
        bogus_idx = 1.5
        assert_raises_regex(
            TypeError,
            f'^usecols must be.*{type(bogus_idx).__name__}',
            np.loadtxt, c, usecols=bogus_idx
            )

        assert_raises_regex(
            TypeError,
            f'^usecols must be.*{type(bogus_idx).__name__}',
            np.loadtxt, c, usecols=[0, bogus_idx, 0]
            )

    def test_bad_usecols(self):
        with pytest.raises(OverflowError):
            np.loadtxt(["1\n"], usecols=[2**64], delimiter=",")
        with pytest.raises((ValueError, OverflowError)):
            # Overflow error on 32bit platforms
            np.loadtxt(["1\n"], usecols=[2**62], delimiter=",")
        with pytest.raises(TypeError,
                match="If a structured dtype .*. But 1 usecols were given and "
                      "the number of fields is 3."):
            np.loadtxt(["1,1\n"], dtype="i,2i", usecols=[0], delimiter=",")

    def test_fancy_dtype(self):
        c = TextIO()
        c.write('1,2,3.0\n4,5,6.0\n')
        c.seek(0)
        dt = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        x = np.loadtxt(c, dtype=dt, delimiter=',')
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dt)
        assert_array_equal(x, a)

    def test_shaped_dtype(self):
        c = TextIO("aaaa  1.0  8.0  1 2 3 4 5 6")
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 3))])
        x = np.loadtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_3d_shaped_dtype(self):
        c = TextIO("aaaa  1.0  8.0  1 2 3 4 5 6 7 8 9 10 11 12")
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 2, 3))])
        x = np.loadtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0,
                       [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_str_dtype(self):
        # see gh-8033
        c = ["str1", "str2"]

        for dt in (str, np.bytes_):
            a = np.array(["str1", "str2"], dtype=dt)
            x = np.loadtxt(c, dtype=dt)
            assert_array_equal(x, a)

    def test_empty_file(self):
        with pytest.warns(UserWarning, match="input contained no data"):
            c = TextIO()
            x = np.loadtxt(c)
            assert_equal(x.shape, (0,))
            x = np.loadtxt(c, dtype=np.int64)
            assert_equal(x.shape, (0,))
            assert_(x.dtype == np.int64)

    def test_unused_converter(self):
        c = TextIO()
        c.writelines(['1 21\n', '3 42\n'])
        c.seek(0)
        data = np.loadtxt(c, usecols=(1,),
                          converters={0: lambda s: int(s, 16)})
        assert_array_equal(data, [21, 42])

        c.seek(0)
        data = np.loadtxt(c, usecols=(1,),
                          converters={1: lambda s: int(s, 16)})
        assert_array_equal(data, [33, 66])

    def test_dtype_with_object(self):
        # Test using an explicit dtype with an object
        data = """ 1; 2001-01-01
                   2; 2002-01-31 """
        ndtype = [('idx', int), ('code', object)]
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        test = np.loadtxt(TextIO(data), delimiter=";", dtype=ndtype,
                          converters=converters)
        control = np.array(
            [(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))],
            dtype=ndtype)
        assert_equal(test, control)

    def test_uint64_type(self):
        tgt = (9223372043271415339, 9223372043271415853)
        c = TextIO()
        c.write("%s %s" % tgt)
        c.seek(0)
        res = np.loadtxt(c, dtype=np.uint64)
        assert_equal(res, tgt)

    def test_int64_type(self):
        tgt = (-9223372036854775807, 9223372036854775807)
        c = TextIO()
        c.write("%s %s" % tgt)
        c.seek(0)
        res = np.loadtxt(c, dtype=np.int64)
        assert_equal(res, tgt)

    def test_from_float_hex(self):
        # IEEE doubles and floats only, otherwise the float32
        # conversion may fail.
        tgt = np.logspace(-10, 10, 5).astype(np.float32)
        tgt = np.hstack((tgt, -tgt)).astype(float)
        inp = '\n'.join(map(float.hex, tgt))
        c = TextIO()
        c.write(inp)
        for dt in [float, np.float32]:
            c.seek(0)
            res = np.loadtxt(
                c, dtype=dt, converters=float.fromhex, encoding="latin1")
            assert_equal(res, tgt, err_msg=f"{dt}")

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    def test_default_float_converter_no_default_hex_conversion(self):
        """
        Ensure that fromhex is only used for values with the correct prefix and
        is not called by default. Regression test related to gh-19598.
        """
        c = TextIO("a b c")
        with pytest.raises(ValueError,
                match=".*convert string 'a' to float64 at row 0, column 1"):
            np.loadtxt(c)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    def test_default_float_converter_exception(self):
        """
        Ensure that the exception message raised during failed floating point
        conversion is correct. Regression test related to gh-19598.
        """
        c = TextIO("qrs tuv")  # Invalid values for default float converter
        with pytest.raises(ValueError,
                match="could not convert string 'qrs' to float64"):
            np.loadtxt(c)

    def test_from_complex(self):
        tgt = (complex(1, 1), complex(1, -1))
        c = TextIO()
        c.write("%s %s" % tgt)
        c.seek(0)
        res = np.loadtxt(c, dtype=complex)
        assert_equal(res, tgt)

    def test_complex_misformatted(self):
        # test for backward compatibility
        # some complex formats used to generate x+-yj
        a = np.zeros((2, 2), dtype=np.complex128)
        re = np.pi
        im = np.e
        a[:] = re - 1.0j * im
        c = BytesIO()
        np.savetxt(c, a, fmt='%.16e')
        c.seek(0)
        txt = c.read()
        c.seek(0)
        # misformat the sign on the imaginary part, gh 7895
        txt_bad = txt.replace(b'e+00-', b'e00+-')
        assert_(txt_bad != txt)
        c.write(txt_bad)
        c.seek(0)
        res = np.loadtxt(c, dtype=complex)
        assert_equal(res, a)

    def test_universal_newline(self):
        with temppath() as name:
            with open(name, 'w') as f:
                f.write('1 21\r3 42\r')
            data = np.loadtxt(name)
        assert_array_equal(data, [[1, 21], [3, 42]])

    def test_empty_field_after_tab(self):
        c = TextIO()
        c.write('1 \t2 \t3\tstart \n4\t5\t6\t  \n7\t8\t9.5\t')
        c.seek(0)
        dt = {'names': ('x', 'y', 'z', 'comment'),
              'formats': ('<i4', '<i4', '<f4', '|S8')}
        x = np.loadtxt(c, dtype=dt, delimiter='\t')
        a = np.array([b'start ', b'  ', b''])
        assert_array_equal(x['comment'], a)

    def test_unpack_structured(self):
        txt = TextIO("M 21 72\nF 35 58")
        dt = {'names': ('a', 'b', 'c'), 'formats': ('|S1', '<i4', '<f4')}
        a, b, c = np.loadtxt(txt, dtype=dt, unpack=True)
        assert_(a.dtype.str == '|S1')
        assert_(b.dtype.str == '<i4')
        assert_(c.dtype.str == '<f4')
        assert_array_equal(a, np.array([b'M', b'F']))
        assert_array_equal(b, np.array([21, 35]))
        assert_array_equal(c, np.array([72.,  58.]))

    def test_ndmin_keyword(self):
        c = TextIO()
        c.write('1,2,3\n4,5,6')
        c.seek(0)
        assert_raises(ValueError, np.loadtxt, c, ndmin=3)
        c.seek(0)
        assert_raises(ValueError, np.loadtxt, c, ndmin=1.5)
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', ndmin=1)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        assert_array_equal(x, a)

        d = TextIO()
        d.write('0,1,2')
        d.seek(0)
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=2)
        assert_(x.shape == (1, 3))
        d.seek(0)
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=1)
        assert_(x.shape == (3,))
        d.seek(0)
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=0)
        assert_(x.shape == (3,))

        e = TextIO()
        e.write('0\n1\n2')
        e.seek(0)
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=2)
        assert_(x.shape == (3, 1))
        e.seek(0)
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=1)
        assert_(x.shape == (3,))
        e.seek(0)
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=0)
        assert_(x.shape == (3,))

        # Test ndmin kw with empty file.
        with pytest.warns(UserWarning, match="input contained no data"):
            f = TextIO()
            assert_(np.loadtxt(f, ndmin=2).shape == (0, 1,))
            assert_(np.loadtxt(f, ndmin=1).shape == (0,))

    def test_generator_source(self):
        def count():
            for i in range(10):
                yield "%d" % i

        res = np.loadtxt(count())
        assert_array_equal(res, np.arange(10))

    def test_bad_line(self):
        c = TextIO()
        c.write('1 2 3\n4 5 6\n2 3')
        c.seek(0)

        # Check for exception and that exception contains line number
        assert_raises_regex(ValueError, "3", np.loadtxt, c)

    def test_none_as_string(self):
        # gh-5155, None should work as string when format demands it
        c = TextIO()
        c.write('100,foo,200\n300,None,400')
        c.seek(0)
        dt = np.dtype([('x', int), ('a', 'S10'), ('y', int)])
        np.loadtxt(c, delimiter=',', dtype=dt, comments=None)  # Should succeed

    @pytest.mark.skipif(locale.getpreferredencoding() == 'ANSI_X3.4-1968',
                        reason="Wrong preferred encoding")
    def test_binary_load(self):
        butf8 = b"5,6,7,\xc3\x95scarscar\r\n15,2,3,hello\r\n"\
                b"20,2,3,\xc3\x95scar\r\n"
        sutf8 = butf8.decode("UTF-8").replace("\r", "").splitlines()
        with temppath() as path:
            with open(path, "wb") as f:
                f.write(butf8)
            with open(path, "rb") as f:
                x = np.loadtxt(f, encoding="UTF-8", dtype=np.str_)
            assert_array_equal(x, sutf8)
            # test broken latin1 conversion people now rely on
            with open(path, "rb") as f:
                x = np.loadtxt(f, encoding="UTF-8", dtype="S")
            x = [b'5,6,7,\xc3\x95scarscar', b'15,2,3,hello', b'20,2,3,\xc3\x95scar']
            assert_array_equal(x, np.array(x, dtype="S"))

    def test_max_rows(self):
        c = TextIO()
        c.write('1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       max_rows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_max_rows_with_skiprows(self):
        c = TextIO()
        c.write('comments\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1, max_rows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

        c = TextIO()
        c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1, max_rows=2)
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8]], int)
        assert_array_equal(x, a)

    def test_max_rows_with_read_continuation(self):
        c = TextIO()
        c.write('1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       max_rows=2)
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8]], int)
        assert_array_equal(x, a)
        # test continuation
        x = np.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([2, 1, 4, 5], int)
        assert_array_equal(x, a)

    def test_max_rows_larger(self):
        #test max_rows > num rows
        c = TextIO()
        c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1, max_rows=6)
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8], [2, 1, 4, 5]], int)
        assert_array_equal(x, a)

    @pytest.mark.parametrize(["skip", "data"], [
            (1, ["ignored\n", "1,2\n", "\n", "3,4\n"]),
            # "Bad" lines that do not end in newlines:
            (1, ["ignored", "1,2", "", "3,4"]),
            (1, StringIO("ignored\n1,2\n\n3,4")),
            # Same as above, but do not skip any lines:
            (0, ["-1,0\n", "1,2\n", "\n", "3,4\n"]),
            (0, ["-1,0", "1,2", "", "3,4"]),
            (0, StringIO("-1,0\n1,2\n\n3,4"))])
    def test_max_rows_empty_lines(self, skip, data):
        with pytest.warns(UserWarning,
                    match=f"Input line 3.*max_rows={3 - skip}"):
            res = np.loadtxt(data, dtype=int, skiprows=skip, delimiter=",",
                             max_rows=3 - skip)
            assert_array_equal(res, [[-1, 0], [1, 2], [3, 4]][skip:])

        if isinstance(data, StringIO):
            data.seek(0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(UserWarning):
                np.loadtxt(data, dtype=int, skiprows=skip, delimiter=",",
                           max_rows=3 - skip)

class Testfromregex:
    def test_record(self):
        c = TextIO()
        c.write('1.312 foo\n1.534 bar\n4.444 qux')
        c.seek(0)

        dt = [('num', np.float64), ('val', 'S3')]
        x = np.fromregex(c, r"([0-9.]+)\s+(...)", dt)
        a = np.array([(1.312, 'foo'), (1.534, 'bar'), (4.444, 'qux')],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_record_2(self):
        c = TextIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)

        dt = [('num', np.int32), ('val', 'S3')]
        x = np.fromregex(c, r"(\d+)\s+(...)", dt)
        a = np.array([(1312, 'foo'), (1534, 'bar'), (4444, 'qux')],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_record_3(self):
        c = TextIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)

        dt = [('num', np.float64)]
        x = np.fromregex(c, r"(\d+)\s+...", dt)
        a = np.array([(1312,), (1534,), (4444,)], dtype=dt)
        assert_array_equal(x, a)

    @pytest.mark.parametrize("path_type", [str, Path])
    def test_record_unicode(self, path_type):
        utf8 = b'\xcf\x96'
        with temppath() as str_path:
            path = path_type(str_path)
            with open(path, 'wb') as f:
                f.write(b'1.312 foo' + utf8 + b' \n1.534 bar\n4.444 qux')

            dt = [('num', np.float64), ('val', 'U4')]
            x = np.fromregex(path, r"(?u)([0-9.]+)\s+(\w+)", dt, encoding='UTF-8')
            a = np.array([(1.312, 'foo' + utf8.decode('UTF-8')), (1.534, 'bar'),
                           (4.444, 'qux')], dtype=dt)
            assert_array_equal(x, a)

            regexp = re.compile(r"([0-9.]+)\s+(\w+)", re.UNICODE)
            x = np.fromregex(path, regexp, dt, encoding='UTF-8')
            assert_array_equal(x, a)

    def test_compiled_bytes(self):
        regexp = re.compile(br'(\d)')
        c = BytesIO(b'123')
        dt = [('num', np.float64)]
        a = np.array([1, 2, 3], dtype=dt)
        x = np.fromregex(c, regexp, dt)
        assert_array_equal(x, a)

    def test_bad_dtype_not_structured(self):
        regexp = re.compile(br'(\d)')
        c = BytesIO(b'123')
        with pytest.raises(TypeError, match='structured datatype'):
            np.fromregex(c, regexp, dtype=np.float64)


#####--------------------------------------------------------------------------


class TestFromTxt(LoadTxtBase):
    loadfunc = staticmethod(np.genfromtxt)

    def test_record(self):
        # Test w/ explicit dtype
        data = TextIO('1 2\n3 4')
        test = np.genfromtxt(data, dtype=[('x', np.int32), ('y', np.int32)])
        control = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_equal(test, control)
        #
        data = TextIO('M 64.0 75.0\nF 25.0 60.0')
        descriptor = {'names': ('gender', 'age', 'weight'),
                      'formats': ('S1', 'i4', 'f4')}
        control = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)],
                           dtype=descriptor)
        test = np.genfromtxt(data, dtype=descriptor)
        assert_equal(test, control)

    def test_array(self):
        # Test outputting a standard ndarray
        data = TextIO('1 2\n3 4')
        control = np.array([[1, 2], [3, 4]], dtype=int)
        test = np.genfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        #
        data.seek(0)
        control = np.array([[1, 2], [3, 4]], dtype=float)
        test = np.loadtxt(data, dtype=float)
        assert_array_equal(test, control)

    def test_1D(self):
        # Test squeezing to 1D
        control = np.array([1, 2, 3, 4], int)
        #
        data = TextIO('1\n2\n3\n4\n')
        test = np.genfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        #
        data = TextIO('1,2,3,4\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',')
        assert_array_equal(test, control)

    def test_comments(self):
        # Test the stripping of comments
        control = np.array([1, 2, 3, 5], int)
        # Comment on its own line
        data = TextIO('# comment\n1,2,3,5\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',', comments='#')
        assert_equal(test, control)
        # Comment at the end of a line
        data = TextIO('1,2,3,5# comment\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',', comments='#')
        assert_equal(test, control)

    def test_skiprows(self):
        # Test row skipping
        control = np.array([1, 2, 3, 5], int)
        kwargs = {"dtype": int, "delimiter": ','}
        #
        data = TextIO('comment\n1,2,3,5\n')
        test = np.genfromtxt(data, skip_header=1, **kwargs)
        assert_equal(test, control)
        #
        data = TextIO('# comment\n1,2,3,5\n')
        test = np.loadtxt(data, skiprows=1, **kwargs)
        assert_equal(test, control)

    def test_skip_footer(self):
        data = [f"# {i}" for i in range(1, 6)]
        data.append("A, B, C")
        data.extend([f"{i},{i:3.1f},{i:03d}" for i in range(51)])
        data[-1] = "99,99"
        kwargs = {"delimiter": ",", "names": True, "skip_header": 5, "skip_footer": 10}
        test = np.genfromtxt(TextIO("\n".join(data)), **kwargs)
        ctrl = np.array([(f"{i:f}", f"{i:f}", f"{i:f}") for i in range(41)],
                        dtype=[(_, float) for _ in "ABC"])
        assert_equal(test, ctrl)

    def test_skip_footer_with_invalid(self):
        with suppress_warnings() as sup:
            sup.filter(ConversionWarning)
            basestr = '1 1\n2 2\n3 3\n4 4\n5  \n6  \n7  \n'
            # Footer too small to get rid of all invalid values
            assert_raises(ValueError, np.genfromtxt,
                          TextIO(basestr), skip_footer=1)
    #        except ValueError:
    #            pass
            a = np.genfromtxt(
                TextIO(basestr), skip_footer=1, invalid_raise=False)
            assert_equal(a, np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]]))
            #
            a = np.genfromtxt(TextIO(basestr), skip_footer=3)
            assert_equal(a, np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]]))
            #
            basestr = '1 1\n2  \n3 3\n4 4\n5  \n6 6\n7 7\n'
            a = np.genfromtxt(
                TextIO(basestr), skip_footer=1, invalid_raise=False)
            assert_equal(a, np.array([[1., 1.], [3., 3.], [4., 4.], [6., 6.]]))
            a = np.genfromtxt(
                TextIO(basestr), skip_footer=3, invalid_raise=False)
            assert_equal(a, np.array([[1., 1.], [3., 3.], [4., 4.]]))

    def test_header(self):
        # Test retrieving a header
        data = TextIO('gender age weight\nM 64.0 75.0\nF 25.0 60.0')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(data, dtype=None, names=True,
                                 encoding='bytes')
            assert_(w[0].category is VisibleDeprecationWarning)
        control = {'gender': np.array([b'M', b'F']),
                   'age': np.array([64.0, 25.0]),
                   'weight': np.array([75.0, 60.0])}
        assert_equal(test['gender'], control['gender'])
        assert_equal(test['age'], control['age'])
        assert_equal(test['weight'], control['weight'])

    def test_auto_dtype(self):
        # Test the automatic definition of the output dtype
        data = TextIO('A 64 75.0 3+4j True\nBCD 25 60.0 5+6j False')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(data, dtype=None, encoding='bytes')
            assert_(w[0].category is VisibleDeprecationWarning)
        control = [np.array([b'A', b'BCD']),
                   np.array([64, 25]),
                   np.array([75.0, 60.0]),
                   np.array([3 + 4j, 5 + 6j]),
                   np.array([True, False]), ]
        assert_equal(test.dtype.names, ['f0', 'f1', 'f2', 'f3', 'f4'])
        for (i, ctrl) in enumerate(control):
            assert_equal(test[f'f{i}'], ctrl)

    def test_auto_dtype_uniform(self):
        # Tests whether the output dtype can be uniformized
        data = TextIO('1 2 3 4\n5 6 7 8\n')
        test = np.genfromtxt(data, dtype=None)
        control = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert_equal(test, control)

    def test_fancy_dtype(self):
        # Check that a nested dtype isn't MIA
        data = TextIO('1,2,3.0\n4,5,6.0\n')
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = np.genfromtxt(data, dtype=fancydtype, delimiter=',')
        control = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=fancydtype)
        assert_equal(test, control)

    def test_names_overwrite(self):
        # Test overwriting the names of the dtype
        descriptor = {'names': ('g', 'a', 'w'),
                      'formats': ('S1', 'i4', 'f4')}
        data = TextIO(b'M 64.0 75.0\nF 25.0 60.0')
        names = ('gender', 'age', 'weight')
        test = np.genfromtxt(data, dtype=descriptor, names=names)
        descriptor['names'] = names
        control = np.array([('M', 64.0, 75.0),
                            ('F', 25.0, 60.0)], dtype=descriptor)
        assert_equal(test, control)

    def test_bad_fname(self):
        with pytest.raises(TypeError, match='fname must be a string,'):
            np.genfromtxt(123)

    def test_commented_header(self):
        # Check that names can be retrieved even if the line is commented out.
        data = TextIO("""
#gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        # The # is part of the first name and should be deleted automatically.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(data, names=True, dtype=None,
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        ctrl = np.array([('M', 21, 72.1), ('F', 35, 58.33), ('M', 33, 21.99)],
                        dtype=[('gender', '|S1'), ('age', int), ('weight', float)])
        assert_equal(test, ctrl)
        # Ditto, but we should get rid of the first element
        data = TextIO(b"""
# gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(data, names=True, dtype=None,
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        assert_equal(test, ctrl)

    def test_names_and_comments_none(self):
        # Tests case when names is true but comments is None (gh-10780)
        data = TextIO('col1 col2\n 1 2\n 3 4')
        test = np.genfromtxt(data, dtype=(int, int), comments=None, names=True)
        control = np.array([(1, 2), (3, 4)], dtype=[('col1', int), ('col2', int)])
        assert_equal(test, control)

    def test_file_is_closed_on_error(self):
        # gh-13200
        with tempdir() as tmpdir:
            fpath = os.path.join(tmpdir, "test.csv")
            with open(fpath, "wb") as f:
                f.write('\N{GREEK PI SYMBOL}'.encode())

            # ResourceWarnings are emitted from a destructor, so won't be
            # detected by regular propagation to errors.
            with assert_no_warnings():
                with pytest.raises(UnicodeDecodeError):
                    np.genfromtxt(fpath, encoding="ascii")

    def test_autonames_and_usecols(self):
        # Tests names and usecols
        data = TextIO('A B C D\n aaaa 121 45 9.1')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(data, usecols=('A', 'C', 'D'),
                                names=True, dtype=None, encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        control = np.array(('aaaa', 45, 9.1),
                           dtype=[('A', '|S4'), ('C', int), ('D', float)])
        assert_equal(test, control)

    def test_converters_with_usecols(self):
        # Test the combination user-defined converters and usecol
        data = TextIO('1,2,3,,5\n6,7,8,9,10\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',',
                            converters={3: lambda s: int(s or - 999)},
                            usecols=(1, 3,))
        control = np.array([[2, -999], [7, 9]], int)
        assert_equal(test, control)

    def test_converters_with_usecols_and_names(self):
        # Tests names and usecols
        data = TextIO('A B C D\n aaaa 121 45 9.1')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(data, usecols=('A', 'C', 'D'), names=True,
                                dtype=None, encoding="bytes",
                                converters={'C': lambda s: 2 * int(s)})
            assert_(w[0].category is VisibleDeprecationWarning)
        control = np.array(('aaaa', 90, 9.1),
                           dtype=[('A', '|S4'), ('C', int), ('D', float)])
        assert_equal(test, control)

    def test_converters_cornercases(self):
        # Test the conversion to datetime.
        converter = {
            'date': lambda s: strptime(s, '%Y-%m-%d %H:%M:%SZ')}
        data = TextIO('2009-02-03 12:00:00Z, 72214.0')
        test = np.genfromtxt(data, delimiter=',', dtype=None,
                            names=['date', 'stid'], converters=converter)
        control = np.array((datetime(2009, 2, 3), 72214.),
                           dtype=[('date', np.object_), ('stid', float)])
        assert_equal(test, control)

    def test_converters_cornercases2(self):
        # Test the conversion to datetime64.
        converter = {
            'date': lambda s: np.datetime64(strptime(s, '%Y-%m-%d %H:%M:%SZ'))}
        data = TextIO('2009-02-03 12:00:00Z, 72214.0')
        test = np.genfromtxt(data, delimiter=',', dtype=None,
                            names=['date', 'stid'], converters=converter)
        control = np.array((datetime(2009, 2, 3), 72214.),
                           dtype=[('date', 'datetime64[us]'), ('stid', float)])
        assert_equal(test, control)

    def test_unused_converter(self):
        # Test whether unused converters are forgotten
        data = TextIO("1 21\n  3 42\n")
        test = np.genfromtxt(data, usecols=(1,),
                            converters={0: lambda s: int(s, 16)})
        assert_equal(test, [21, 42])
        #
        data.seek(0)
        test = np.genfromtxt(data, usecols=(1,),
                            converters={1: lambda s: int(s, 16)})
        assert_equal(test, [33, 66])

    def test_invalid_converter(self):
        strip_rand = lambda x: float((b'r' in x.lower() and x.split()[-1]) or
                                     ((b'r' not in x.lower() and x.strip()) or 0.0))
        strip_per = lambda x: float((b'%' in x.lower() and x.split()[0]) or
                                    ((b'%' not in x.lower() and x.strip()) or 0.0))
        s = TextIO("D01N01,10/1/2003 ,1 %,R 75,400,600\r\n"
                   "L24U05,12/5/2003, 2 %,1,300, 150.5\r\n"
                   "D02N03,10/10/2004,R 1,,7,145.55")
        kwargs = {
            "converters": {2: strip_per, 3: strip_rand}, "delimiter": ",",
            "dtype": None, "encoding": "bytes"}
        assert_raises(ConverterError, np.genfromtxt, s, **kwargs)

    def test_tricky_converter_bug1666(self):
        # Test some corner cases
        s = TextIO('q1,2\nq3,4')
        cnv = lambda s: float(s[1:])
        test = np.genfromtxt(s, delimiter=',', converters={0: cnv})
        control = np.array([[1., 2.], [3., 4.]])
        assert_equal(test, control)

    def test_dtype_with_converters(self):
        dstr = "2009; 23; 46"
        test = np.genfromtxt(TextIO(dstr,),
                            delimiter=";", dtype=float, converters={0: bytes})
        control = np.array([('2009', 23., 46)],
                           dtype=[('f0', '|S4'), ('f1', float), ('f2', float)])
        assert_equal(test, control)
        test = np.genfromtxt(TextIO(dstr,),
                            delimiter=";", dtype=float, converters={0: float})
        control = np.array([2009., 23., 46],)
        assert_equal(test, control)

    @pytest.mark.filterwarnings("ignore:.*recfromcsv.*:DeprecationWarning")
    def test_dtype_with_converters_and_usecols(self):
        dstr = "1,5,-1,1:1\n2,8,-1,1:n\n3,3,-2,m:n\n"
        dmap = {'1:1': 0, '1:n': 1, 'm:1': 2, 'm:n': 3}
        dtyp = [('e1', 'i4'), ('e2', 'i4'), ('e3', 'i2'), ('n', 'i1')]
        conv = {0: int, 1: int, 2: int, 3: lambda r: dmap[r.decode()]}
        test = recfromcsv(TextIO(dstr,), dtype=dtyp, delimiter=',',
                          names=None, converters=conv, encoding="bytes")
        control = np.rec.array([(1, 5, -1, 0), (2, 8, -1, 1), (3, 3, -2, 3)], dtype=dtyp)
        assert_equal(test, control)
        dtyp = [('e1', 'i4'), ('e2', 'i4'), ('n', 'i1')]
        test = recfromcsv(TextIO(dstr,), dtype=dtyp, delimiter=',',
                          usecols=(0, 1, 3), names=None, converters=conv,
                          encoding="bytes")
        control = np.rec.array([(1, 5, 0), (2, 8, 1), (3, 3, 3)], dtype=dtyp)
        assert_equal(test, control)

    def test_dtype_with_object(self):
        # Test using an explicit dtype with an object
        data = """ 1; 2001-01-01
                   2; 2002-01-31 """
        ndtype = [('idx', int), ('code', object)]
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        test = np.genfromtxt(TextIO(data), delimiter=";", dtype=ndtype,
                             converters=converters)
        control = np.array(
            [(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))],
            dtype=ndtype)
        assert_equal(test, control)

        ndtype = [('nest', [('idx', int), ('code', object)])]
        with assert_raises_regex(NotImplementedError,
                                 'Nested fields.* not supported.*'):
            test = np.genfromtxt(TextIO(data), delimiter=";",
                                 dtype=ndtype, converters=converters)

        # nested but empty fields also aren't supported
        ndtype = [('idx', int), ('code', object), ('nest', [])]
        with assert_raises_regex(NotImplementedError,
                                 'Nested fields.* not supported.*'):
            test = np.genfromtxt(TextIO(data), delimiter=";",
                                 dtype=ndtype, converters=converters)

    def test_dtype_with_object_no_converter(self):
        # Object without a converter uses bytes:
        parsed = np.genfromtxt(TextIO("1"), dtype=object)
        assert parsed[()] == b"1"
        parsed = np.genfromtxt(TextIO("string"), dtype=object)
        assert parsed[()] == b"string"

    def test_userconverters_with_explicit_dtype(self):
        # Test user_converters w/ explicit (standard) dtype
        data = TextIO('skip,skip,2001-01-01,1.0,skip')
        test = np.genfromtxt(data, delimiter=",", names=None, dtype=float,
                             usecols=(2, 3), converters={2: bytes})
        control = np.array([('2001-01-01', 1.)],
                           dtype=[('', '|S10'), ('', float)])
        assert_equal(test, control)

    def test_utf8_userconverters_with_explicit_dtype(self):
        utf8 = b'\xcf\x96'
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write(b'skip,skip,2001-01-01' + utf8 + b',1.0,skip')
            test = np.genfromtxt(path, delimiter=",", names=None, dtype=float,
                                 usecols=(2, 3), converters={2: str},
                                 encoding='UTF-8')
        control = np.array([('2001-01-01' + utf8.decode('UTF-8'), 1.)],
                           dtype=[('', '|U11'), ('', float)])
        assert_equal(test, control)

    def test_spacedelimiter(self):
        # Test space delimiter
        data = TextIO("1  2  3  4   5\n6  7  8  9  10")
        test = np.genfromtxt(data)
        control = np.array([[1., 2., 3., 4., 5.],
                            [6., 7., 8., 9., 10.]])
        assert_equal(test, control)

    def test_integer_delimiter(self):
        # Test using an integer for delimiter
        data = "  1  2  3\n  4  5 67\n890123  4"
        test = np.genfromtxt(TextIO(data), delimiter=3)
        control = np.array([[1, 2, 3], [4, 5, 67], [890, 123, 4]])
        assert_equal(test, control)

    def test_missing(self):
        data = TextIO('1,2,3,,5\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',',
                            converters={3: lambda s: int(s or - 999)})
        control = np.array([1, 2, 3, -999, 5], int)
        assert_equal(test, control)

    def test_missing_with_tabs(self):
        # Test w/ a delimiter tab
        txt = "1\t2\t3\n\t2\t\n1\t\t3"
        test = np.genfromtxt(TextIO(txt), delimiter="\t",
                             usemask=True,)
        ctrl_d = np.array([(1, 2, 3), (np.nan, 2, np.nan), (1, np.nan, 3)],)
        ctrl_m = np.array([(0, 0, 0), (1, 0, 1), (0, 1, 0)], dtype=bool)
        assert_equal(test.data, ctrl_d)
        assert_equal(test.mask, ctrl_m)

    def test_usecols(self):
        # Test the selection of columns
        # Select 1 column
        control = np.array([[1, 2], [3, 4]], float)
        data = TextIO()
        np.savetxt(data, control)
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=(1,))
        assert_equal(test, control[:, 1])
        #
        control = np.array([[1, 2, 3], [3, 4, 5]], float)
        data = TextIO()
        np.savetxt(data, control)
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=(1, 2))
        assert_equal(test, control[:, 1:])
        # Testing with arrays instead of tuples.
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=np.array([1, 2]))
        assert_equal(test, control[:, 1:])

    def test_usecols_as_css(self):
        # Test giving usecols with a comma-separated string
        data = "1 2 3\n4 5 6"
        test = np.genfromtxt(TextIO(data),
                             names="a, b, c", usecols="a, c")
        ctrl = np.array([(1, 3), (4, 6)], dtype=[(_, float) for _ in "ac"])
        assert_equal(test, ctrl)

    def test_usecols_with_structured_dtype(self):
        # Test usecols with an explicit structured dtype
        data = TextIO("JOE 70.1 25.3\nBOB 60.5 27.9")
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        test = np.genfromtxt(
            data, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        assert_equal(test['stid'], [b"JOE", b"BOB"])
        assert_equal(test['temp'], [25.3, 27.9])

    def test_usecols_with_integer(self):
        # Test usecols with an integer
        test = np.genfromtxt(TextIO(b"1 2 3\n4 5 6"), usecols=0)
        assert_equal(test, np.array([1., 4.]))

    def test_usecols_with_named_columns(self):
        # Test usecols with named columns
        ctrl = np.array([(1, 3), (4, 6)], dtype=[('a', float), ('c', float)])
        data = "1 2 3\n4 5 6"
        kwargs = {"names": "a, b, c"}
        test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(data),
                             usecols=('a', 'c'), **kwargs)
        assert_equal(test, ctrl)

    def test_empty_file(self):
        # Test that an empty file raises the proper warning.
        with suppress_warnings() as sup:
            sup.filter(message="genfromtxt: Empty input file:")
            data = TextIO()
            test = np.genfromtxt(data)
            assert_equal(test, np.array([]))

            # when skip_header > 0
            test = np.genfromtxt(data, skip_header=1)
            assert_equal(test, np.array([]))

    def test_fancy_dtype_alt(self):
        # Check that a nested dtype isn't MIA
        data = TextIO('1,2,3.0\n4,5,6.0\n')
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = np.genfromtxt(data, dtype=fancydtype, delimiter=',', usemask=True)
        control = ma.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=fancydtype)
        assert_equal(test, control)

    def test_shaped_dtype(self):
        c = TextIO("aaaa  1.0  8.0  1 2 3 4 5 6")
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 3))])
        x = np.genfromtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_withmissing(self):
        data = TextIO('A,B\n0,1\n2,N/A')
        kwargs = {"delimiter": ",", "missing_values": "N/A", "names": True}
        test = np.genfromtxt(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        #
        data.seek(0)
        test = np.genfromtxt(data, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', float), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_user_missing_values(self):
        data = "A, B, C\n0, 0., 0j\n1, N/A, 1j\n-9, 2.2, N/A\n3, -99, 3j"
        basekwargs = {"dtype": None, "delimiter": ",", "names": True}
        mdtype = [('A', int), ('B', float), ('C', complex)]
        #
        test = np.genfromtxt(TextIO(data), missing_values="N/A",
                            **basekwargs)
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j),
                            (-9, 2.2, -999j), (3, -99, 3j)],
                           mask=[(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
                           dtype=mdtype)
        assert_equal(test, control)
        #
        basekwargs['dtype'] = mdtype
        test = np.genfromtxt(TextIO(data),
                            missing_values={0: -9, 1: -99, 2: -999j}, usemask=True, **basekwargs)
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j),
                            (-9, 2.2, -999j), (3, -99, 3j)],
                           mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)],
                           dtype=mdtype)
        assert_equal(test, control)
        #
        test = np.genfromtxt(TextIO(data),
                            missing_values={0: -9, 'B': -99, 'C': -999j},
                            usemask=True,
                            **basekwargs)
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j),
                            (-9, 2.2, -999j), (3, -99, 3j)],
                           mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)],
                           dtype=mdtype)
        assert_equal(test, control)

    def test_user_filling_values(self):
        # Test with missing and filling values
        ctrl = np.array([(0, 3), (4, -999)], dtype=[('a', int), ('b', int)])
        data = "N/A, 2, 3\n4, ,???"
        kwargs = {"delimiter": ",",
                      "dtype": int,
                      "names": "a,b,c",
                      "missing_values": {0: "N/A", 'b': " ", 2: "???"},
                      "filling_values": {0: 0, 'b': 0, 2: -999}}
        test = np.genfromtxt(TextIO(data), **kwargs)
        ctrl = np.array([(0, 2, 3), (4, 0, -999)],
                        dtype=[(_, int) for _ in "abc"])
        assert_equal(test, ctrl)
        #
        test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
        ctrl = np.array([(0, 3), (4, -999)], dtype=[(_, int) for _ in "ac"])
        assert_equal(test, ctrl)

        data2 = "1,2,*,4\n5,*,7,8\n"
        test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int,
                             missing_values="*", filling_values=0)
        ctrl = np.array([[1, 2, 0, 4], [5, 0, 7, 8]])
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int,
                             missing_values="*", filling_values=-1)
        ctrl = np.array([[1, 2, -1, 4], [5, -1, 7, 8]])
        assert_equal(test, ctrl)

    def test_withmissing_float(self):
        data = TextIO('A,B\n0,1.5\n2,-999.00')
        test = np.genfromtxt(data, dtype=None, delimiter=',',
                            missing_values='-999.0', names=True, usemask=True)
        control = ma.array([(0, 1.5), (2, -1.)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_with_masked_column_uniform(self):
        # Test masked column
        data = TextIO('1 2 3\n4 5 6\n')
        test = np.genfromtxt(data, dtype=None,
                             missing_values='2,5', usemask=True)
        control = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 1, 0]])
        assert_equal(test, control)

    def test_with_masked_column_various(self):
        # Test masked column
        data = TextIO('True 2 3\nFalse 5 6\n')
        test = np.genfromtxt(data, dtype=None,
                             missing_values='2,5', usemask=True)
        control = ma.array([(1, 2, 3), (0, 5, 6)],
                           mask=[(0, 1, 0), (0, 1, 0)],
                           dtype=[('f0', bool), ('f1', bool), ('f2', int)])
        assert_equal(test, control)

    def test_invalid_raise(self):
        # Test invalid raise
        data = ["1, 1, 1, 1, 1"] * 50
        for i in range(5):
            data[10 * i] = "2, 2, 2, 2 2"
        data.insert(0, "a, b, c, d, e")
        mdata = TextIO("\n".join(data))

        kwargs = {"delimiter": ",", "dtype": None, "names": True}

        def f():
            return np.genfromtxt(mdata, invalid_raise=False, **kwargs)
        mtest = assert_warns(ConversionWarning, f)
        assert_equal(len(mtest), 45)
        assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'abcde']))
        #
        mdata.seek(0)
        assert_raises(ValueError, np.genfromtxt, mdata,
                      delimiter=",", names=True)

    def test_invalid_raise_with_usecols(self):
        # Test invalid_raise with usecols
        data = ["1, 1, 1, 1, 1"] * 50
        for i in range(5):
            data[10 * i] = "2, 2, 2, 2 2"
        data.insert(0, "a, b, c, d, e")
        mdata = TextIO("\n".join(data))

        kwargs = {"delimiter": ",", "dtype": None, "names": True,
                      "invalid_raise": False}

        def f():
            return np.genfromtxt(mdata, usecols=(0, 4), **kwargs)
        mtest = assert_warns(ConversionWarning, f)
        assert_equal(len(mtest), 45)
        assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'ae']))
        #
        mdata.seek(0)
        mtest = np.genfromtxt(mdata, usecols=(0, 1), **kwargs)
        assert_equal(len(mtest), 50)
        control = np.ones(50, dtype=[(_, int) for _ in 'ab'])
        control[[10 * _ for _ in range(5)]] = (2, 2)
        assert_equal(mtest, control)

    def test_inconsistent_dtype(self):
        # Test inconsistent dtype
        data = ["1, 1, 1, 1, -1.1"] * 50
        mdata = TextIO("\n".join(data))

        converters = {4: lambda x: f"({x.decode()})"}
        kwargs = {"delimiter": ",", "converters": converters,
                      "dtype": [(_, int) for _ in 'abcde'], "encoding": "bytes"}
        assert_raises(ValueError, np.genfromtxt, mdata, **kwargs)

    def test_default_field_format(self):
        # Test default format
        data = "0, 1, 2.3\n4, 5, 6.7"
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=None, defaultfmt="f%02i")
        ctrl = np.array([(0, 1, 2.3), (4, 5, 6.7)],
                        dtype=[("f00", int), ("f01", int), ("f02", float)])
        assert_equal(mtest, ctrl)

    def test_single_dtype_wo_names(self):
        # Test single dtype w/o names
        data = "0, 1, 2.3\n4, 5, 6.7"
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=float, defaultfmt="f%02i")
        ctrl = np.array([[0., 1., 2.3], [4., 5., 6.7]], dtype=float)
        assert_equal(mtest, ctrl)

    def test_single_dtype_w_explicit_names(self):
        # Test single dtype w explicit names
        data = "0, 1, 2.3\n4, 5, 6.7"
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=float, names="a, b, c")
        ctrl = np.array([(0., 1., 2.3), (4., 5., 6.7)],
                        dtype=[(_, float) for _ in "abc"])
        assert_equal(mtest, ctrl)

    def test_single_dtype_w_implicit_names(self):
        # Test single dtype w implicit names
        data = "a, b, c\n0, 1, 2.3\n4, 5, 6.7"
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=float, names=True)
        ctrl = np.array([(0., 1., 2.3), (4., 5., 6.7)],
                        dtype=[(_, float) for _ in "abc"])
        assert_equal(mtest, ctrl)

    def test_easy_structured_dtype(self):
        # Test easy structured dtype
        data = "0, 1, 2.3\n4, 5, 6.7"
        mtest = np.genfromtxt(TextIO(data), delimiter=",",
                             dtype=(int, float, float), defaultfmt="f_%02i")
        ctrl = np.array([(0, 1., 2.3), (4, 5., 6.7)],
                        dtype=[("f_00", int), ("f_01", float), ("f_02", float)])
        assert_equal(mtest, ctrl)

    def test_autostrip(self):
        # Test autostrip
        data = "01/01/2003  , 1.3,   abcde"
        kwargs = {"delimiter": ",", "dtype": None, "encoding": "bytes"}
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            mtest = np.genfromtxt(TextIO(data), **kwargs)
            assert_(w[0].category is VisibleDeprecationWarning)
        ctrl = np.array([('01/01/2003  ', 1.3, '   abcde')],
                        dtype=[('f0', '|S12'), ('f1', float), ('f2', '|S8')])
        assert_equal(mtest, ctrl)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            mtest = np.genfromtxt(TextIO(data), autostrip=True, **kwargs)
            assert_(w[0].category is VisibleDeprecationWarning)
        ctrl = np.array([('01/01/2003', 1.3, 'abcde')],
                        dtype=[('f0', '|S10'), ('f1', float), ('f2', '|S5')])
        assert_equal(mtest, ctrl)

    def test_replace_space(self):
        # Test the 'replace_space' option
        txt = "A.A, B (B), C:C\n1, 2, 3.14"
        # Test default: replace ' ' by '_' and delete non-alphanum chars
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=None)
        ctrl_dtype = [("AA", int), ("B_B", int), ("CC", float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # Test: no replace, no delete
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=None,
                             replace_space='', deletechars='')
        ctrl_dtype = [("A.A", int), ("B (B)", int), ("C:C", float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # Test: no delete (spaces are replaced by _)
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=None,
                             deletechars='')
        ctrl_dtype = [("A.A", int), ("B_(B)", int), ("C:C", float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)

    def test_replace_space_known_dtype(self):
        # Test the 'replace_space' (and related) options when dtype != None
        txt = "A.A, B (B), C:C\n1, 2, 3"
        # Test default: replace ' ' by '_' and delete non-alphanum chars
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=int)
        ctrl_dtype = [("AA", int), ("B_B", int), ("CC", int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # Test: no replace, no delete
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=int,
                             replace_space='', deletechars='')
        ctrl_dtype = [("A.A", int), ("B (B)", int), ("C:C", int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # Test: no delete (spaces are replaced by _)
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=int,
                             deletechars='')
        ctrl_dtype = [("A.A", int), ("B_(B)", int), ("C:C", int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)

    def test_incomplete_names(self):
        # Test w/ incomplete names
        data = "A,,C\n0,1,2\n3,4,5"
        kwargs = {"delimiter": ",", "names": True}
        # w/ dtype=None
        ctrl = np.array([(0, 1, 2), (3, 4, 5)],
                        dtype=[(_, int) for _ in ('A', 'f0', 'C')])
        test = np.genfromtxt(TextIO(data), dtype=None, **kwargs)
        assert_equal(test, ctrl)
        # w/ default dtype
        ctrl = np.array([(0, 1, 2), (3, 4, 5)],
                        dtype=[(_, float) for _ in ('A', 'f0', 'C')])
        test = np.genfromtxt(TextIO(data), **kwargs)

    def test_names_auto_completion(self):
        # Make sure that names are properly completed
        data = "1 2 3\n 4 5 6"
        test = np.genfromtxt(TextIO(data),
                             dtype=(int, float, int), names="a")
        ctrl = np.array([(1, 2, 3), (4, 5, 6)],
                        dtype=[('a', int), ('f0', float), ('f1', int)])
        assert_equal(test, ctrl)

    def test_names_with_usecols_bug1636(self):
        # Make sure we pick up the right names w/ usecols
        data = "A,B,C,D,E\n0,1,2,3,4\n0,1,2,3,4\n0,1,2,3,4"
        ctrl_names = ("A", "C", "E")
        test = np.genfromtxt(TextIO(data),
                             dtype=(int, int, int), delimiter=",",
                             usecols=(0, 2, 4), names=True)
        assert_equal(test.dtype.names, ctrl_names)
        #
        test = np.genfromtxt(TextIO(data),
                             dtype=(int, int, int), delimiter=",",
                             usecols=("A", "C", "E"), names=True)
        assert_equal(test.dtype.names, ctrl_names)
        #
        test = np.genfromtxt(TextIO(data),
                             dtype=int, delimiter=",",
                             usecols=("A", "C", "E"), names=True)
        assert_equal(test.dtype.names, ctrl_names)

    def test_fixed_width_names(self):
        # Test fix-width w/ names
        data = "    A    B   C\n    0    1 2.3\n   45   67   9."
        kwargs = {"delimiter": (5, 5, 4), "names": True, "dtype": None}
        ctrl = np.array([(0, 1, 2.3), (45, 67, 9.)],
                        dtype=[('A', int), ('B', int), ('C', float)])
        test = np.genfromtxt(TextIO(data), **kwargs)
        assert_equal(test, ctrl)
        #
        kwargs = {"delimiter": 5, "names": True, "dtype": None}
        ctrl = np.array([(0, 1, 2.3), (45, 67, 9.)],
                        dtype=[('A', int), ('B', int), ('C', float)])
        test = np.genfromtxt(TextIO(data), **kwargs)
        assert_equal(test, ctrl)

    def test_filling_values(self):
        # Test missing values
        data = b"1, 2, 3\n1, , 5\n0, 6, \n"
        kwargs = {"delimiter": ",", "dtype": None, "filling_values": -999}
        ctrl = np.array([[1, 2, 3], [1, -999, 5], [0, 6, -999]], dtype=int)
        test = np.genfromtxt(TextIO(data), **kwargs)
        assert_equal(test, ctrl)

    def test_comments_is_none(self):
        # Github issue 329 (None was previously being converted to 'None').
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO("test1,testNonetherestofthedata"),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        assert_equal(test[1], b'testNonetherestofthedata')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO("test1, testNonetherestofthedata"),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        assert_equal(test[1], b' testNonetherestofthedata')

    def test_latin1(self):
        latin1 = b'\xf6\xfc\xf6'
        norm = b"norm1,norm2,norm3\n"
        enc = b"test1,testNonethe" + latin1 + b",test3\n"
        s = norm + enc + norm
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO(s),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        assert_equal(test[1, 0], b"test1")
        assert_equal(test[1, 1], b"testNonethe" + latin1)
        assert_equal(test[1, 2], b"test3")
        test = np.genfromtxt(TextIO(s),
                             dtype=None, comments=None, delimiter=',',
                             encoding='latin1')
        assert_equal(test[1, 0], "test1")
        assert_equal(test[1, 1], "testNonethe" + latin1.decode('latin1'))
        assert_equal(test[1, 2], "test3")

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO(b"0,testNonethe" + latin1),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        assert_equal(test['f0'], 0)
        assert_equal(test['f1'], b"testNonethe" + latin1)

    def test_binary_decode_autodtype(self):
        utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
        v = self.loadfunc(BytesIO(utf16), dtype=None, encoding='UTF-16')
        assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))

    def test_utf8_byte_encoding(self):
        utf8 = b"\xcf\x96"
        norm = b"norm1,norm2,norm3\n"
        enc = b"test1,testNonethe" + utf8 + b",test3\n"
        s = norm + enc + norm
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO(s),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        ctl = np.array([
                 [b'norm1', b'norm2', b'norm3'],
                 [b'test1', b'testNonethe' + utf8, b'test3'],
                 [b'norm1', b'norm2', b'norm3']])
        assert_array_equal(test, ctl)

    def test_utf8_file(self):
        utf8 = b"\xcf\x96"
        with temppath() as path:
            with open(path, "wb") as f:
                f.write((b"test1,testNonethe" + utf8 + b",test3\n") * 2)
            test = np.genfromtxt(path, dtype=None, comments=None,
                                 delimiter=',', encoding="UTF-8")
            ctl = np.array([
                     ["test1", "testNonethe" + utf8.decode("UTF-8"), "test3"],
                     ["test1", "testNonethe" + utf8.decode("UTF-8"), "test3"]],
                     dtype=np.str_)
            assert_array_equal(test, ctl)

            # test a mixed dtype
            with open(path, "wb") as f:
                f.write(b"0,testNonethe" + utf8)
            test = np.genfromtxt(path, dtype=None, comments=None,
                                 delimiter=',', encoding="UTF-8")
            assert_equal(test['f0'], 0)
            assert_equal(test['f1'], "testNonethe" + utf8.decode("UTF-8"))

    def test_utf8_file_nodtype_unicode(self):
        # bytes encoding with non-latin1 -> unicode upcast
        utf8 = '\u03d6'
        latin1 = '\xf6\xfc\xf6'

        # skip test if cannot encode utf8 test string with preferred
        # encoding. The preferred encoding is assumed to be the default
        # encoding of open. Will need to change this for PyTest, maybe
        # using pytest.mark.xfail(raises=***).
        try:
            encoding = locale.getpreferredencoding()
            utf8.encode(encoding)
        except (UnicodeError, ImportError):
            pytest.skip('Skipping test_utf8_file_nodtype_unicode, '
                        'unable to encode utf8 in preferred encoding')

        with temppath() as path:
            with open(path, "wt") as f:
                f.write("norm1,norm2,norm3\n")
                f.write("norm1," + latin1 + ",norm3\n")
                f.write("test1,testNonethe" + utf8 + ",test3\n")
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always', '',
                                        VisibleDeprecationWarning)
                test = np.genfromtxt(path, dtype=None, comments=None,
                                     delimiter=',', encoding="bytes")
                # Check for warning when encoding not specified.
                assert_(w[0].category is VisibleDeprecationWarning)
            ctl = np.array([
                     ["norm1", "norm2", "norm3"],
                     ["norm1", latin1, "norm3"],
                     ["test1", "testNonethe" + utf8, "test3"]],
                     dtype=np.str_)
            assert_array_equal(test, ctl)

    @pytest.mark.filterwarnings("ignore:.*recfromtxt.*:DeprecationWarning")
    def test_recfromtxt(self):
        #
        data = TextIO('A,B\n0,1\n2,3')
        kwargs = {"delimiter": ",", "missing_values": "N/A", "names": True}
        test = recfromtxt(data, **kwargs)
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('A', int), ('B', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        #
        data = TextIO('A,B\n0,1\n2,N/A')
        test = recfromtxt(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.A, [0, 2])

    @pytest.mark.filterwarnings("ignore:.*recfromcsv.*:DeprecationWarning")
    def test_recfromcsv(self):
        #
        data = TextIO('A,B\n0,1\n2,3')
        kwargs = {"missing_values": "N/A", "names": True, "case_sensitive": True,
                      "encoding": "bytes"}
        test = recfromcsv(data, dtype=None, **kwargs)
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('A', int), ('B', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        #
        data = TextIO('A,B\n0,1\n2,N/A')
        test = recfromcsv(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.A, [0, 2])
        #
        data = TextIO('A,B\n0,1\n2,3')
        test = recfromcsv(data, missing_values='N/A',)
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('a', int), ('b', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        #
        data = TextIO('A,B\n0,1\n2,3')
        dtype = [('a', int), ('b', float)]
        test = recfromcsv(data, missing_values='N/A', dtype=dtype)
        control = np.array([(0, 1), (2, 3)],
                           dtype=dtype)
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)

        # gh-10394
        data = TextIO('color\n"red"\n"blue"')
        test = recfromcsv(data, converters={0: lambda x: x.strip('\"')})
        control = np.array([('red',), ('blue',)], dtype=[('color', (str, 4))])
        assert_equal(test.dtype, control.dtype)
        assert_equal(test, control)

    def test_max_rows(self):
        # Test the `max_rows` keyword argument.
        data = '1 2\n3 4\n5 6\n7 8\n9 10\n'
        txt = TextIO(data)
        a1 = np.genfromtxt(txt, max_rows=3)
        a2 = np.genfromtxt(txt)
        assert_equal(a1, [[1, 2], [3, 4], [5, 6]])
        assert_equal(a2, [[7, 8], [9, 10]])

        # max_rows must be at least 1.
        assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=0)

        # An input with several invalid rows.
        data = '1 1\n2 2\n0 \n3 3\n4 4\n5  \n6  \n7  \n'

        test = np.genfromtxt(TextIO(data), max_rows=2)
        control = np.array([[1., 1.], [2., 2.]])
        assert_equal(test, control)

        # Test keywords conflict
        assert_raises(ValueError, np.genfromtxt, TextIO(data), skip_footer=1,
                      max_rows=4)

        # Test with invalid value
        assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=4)

        # Test with invalid not raise
        with suppress_warnings() as sup:
            sup.filter(ConversionWarning)

            test = np.genfromtxt(TextIO(data), max_rows=4, invalid_raise=False)
            control = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
            assert_equal(test, control)

            test = np.genfromtxt(TextIO(data), max_rows=5, invalid_raise=False)
            control = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
            assert_equal(test, control)

        # Structured array with field names.
        data = 'a b\n#c d\n1 1\n2 2\n#0 \n3 3\n4 4\n5  5\n'

        # Test with header, names and comments
        txt = TextIO(data)
        test = np.genfromtxt(txt, skip_header=1, max_rows=3, names=True)
        control = np.array([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                      dtype=[('c', '<f8'), ('d', '<f8')])
        assert_equal(test, control)
        # To continue reading the same "file", don't use skip_header or
        # names, and use the previously determined dtype.
        test = np.genfromtxt(txt, max_rows=None, dtype=test.dtype)
        control = np.array([(4.0, 4.0), (5.0, 5.0)],
                      dtype=[('c', '<f8'), ('d', '<f8')])
        assert_equal(test, control)

    def test_gft_using_filename(self):
        # Test that we can load data from a filename as well as a file
        # object
        tgt = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')

        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            with temppath() as name:
                with open(name, 'w') as f:
                    f.write(data)
                res = np.genfromtxt(name)
            assert_array_equal(res, tgt)

    def test_gft_from_gzip(self):
        # Test that we can load data from a gzipped file
        wanted = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')

        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            s = BytesIO()
            with gzip.GzipFile(fileobj=s, mode='w') as g:
                g.write(asbytes(data))

            with temppath(suffix='.gz2') as name:
                with open(name, 'w') as f:
                    f.write(data)
                assert_array_equal(np.genfromtxt(name), wanted)

    def test_gft_using_generator(self):
        # gft doesn't work with unicode.
        def count():
            for i in range(10):
                yield asbytes("%d" % i)

        res = np.genfromtxt(count())
        assert_array_equal(res, np.arange(10))

    def test_auto_dtype_largeint(self):
        # Regression test for numpy/numpy#5635 whereby large integers could
        # cause OverflowErrors.

        # Test the automatic definition of the output dtype
        #
        # 2**66 = 73786976294838206464 => should convert to float
        # 2**34 = 17179869184 => should convert to int64
        # 2**10 = 1024 => should convert to int (int32 on 32-bit systems,
        #                 int64 on 64-bit systems)

        data = TextIO('73786976294838206464 17179869184 1024')

        test = np.genfromtxt(data, dtype=None)

        assert_equal(test.dtype.names, ['f0', 'f1', 'f2'])

        assert_(test.dtype['f0'] == float)
        assert_(test.dtype['f1'] == np.int64)
        assert_(test.dtype['f2'] == np.int_)

        assert_allclose(test['f0'], 73786976294838206464.)
        assert_equal(test['f1'], 17179869184)
        assert_equal(test['f2'], 1024)

    def test_unpack_float_data(self):
        txt = TextIO("1,2,3\n4,5,6\n7,8,9\n0.0,1.0,2.0")
        a, b, c = np.loadtxt(txt, delimiter=",", unpack=True)
        assert_array_equal(a, np.array([1.0, 4.0, 7.0, 0.0]))
        assert_array_equal(b, np.array([2.0, 5.0, 8.0, 1.0]))
        assert_array_equal(c, np.array([3.0, 6.0, 9.0, 2.0]))

    def test_unpack_structured(self):
        # Regression test for gh-4341
        # Unpacking should work on structured arrays
        txt = TextIO("M 21 72\nF 35 58")
        dt = {'names': ('a', 'b', 'c'), 'formats': ('S1', 'i4', 'f4')}
        a, b, c = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_equal(a.dtype, np.dtype('S1'))
        assert_equal(b.dtype, np.dtype('i4'))
        assert_equal(c.dtype, np.dtype('f4'))
        assert_array_equal(a, np.array([b'M', b'F']))
        assert_array_equal(b, np.array([21, 35]))
        assert_array_equal(c, np.array([72.,  58.]))

    def test_unpack_auto_dtype(self):
        # Regression test for gh-4341
        # Unpacking should work when dtype=None
        txt = TextIO("M 21 72.\nF 35 58.")
        expected = (np.array(["M", "F"]), np.array([21, 35]), np.array([72., 58.]))
        test = np.genfromtxt(txt, dtype=None, unpack=True, encoding="utf-8")
        for arr, result in zip(expected, test):
            assert_array_equal(arr, result)
            assert_equal(arr.dtype, result.dtype)

    def test_unpack_single_name(self):
        # Regression test for gh-4341
        # Unpacking should work when structured dtype has only one field
        txt = TextIO("21\n35")
        dt = {'names': ('a',), 'formats': ('i4',)}
        expected = np.array([21, 35], dtype=np.int32)
        test = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_array_equal(expected, test)
        assert_equal(expected.dtype, test.dtype)

    def test_squeeze_scalar(self):
        # Regression test for gh-4341
        # Unpacking a scalar should give zero-dim output,
        # even if dtype is structured
        txt = TextIO("1")
        dt = {'names': ('a',), 'formats': ('i4',)}
        expected = np.array((1,), dtype=np.int32)
        test = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_array_equal(expected, test)
        assert_equal((), test.shape)
        assert_equal(expected.dtype, test.dtype)

    @pytest.mark.parametrize("ndim", [0, 1, 2])
    def test_ndmin_keyword(self, ndim: int):
        # lets have the same behaviour of ndmin as loadtxt
        # as they should be the same for non-missing values
        txt = "42"

        a = np.loadtxt(StringIO(txt), ndmin=ndim)
        b = np.genfromtxt(StringIO(txt), ndmin=ndim)

        assert_array_equal(a, b)


class TestPathUsage:
    # Test that pathlib.Path can be used
    def test_loadtxt(self):
        with temppath(suffix='.txt') as path:
            path = Path(path)
            a = np.array([[1.1, 2], [3, 4]])
            np.savetxt(path, a)
            x = np.loadtxt(path)
            assert_array_equal(x, a)

    def test_save_load(self):
        # Test that pathlib.Path instances can be used with save.
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            np.save(path, a)
            data = np.load(path)
            assert_array_equal(data, a)

    def test_save_load_memmap(self):
        # Test that pathlib.Path instances can be loaded mem-mapped.
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            np.save(path, a)
            data = np.load(path, mmap_mode='r')
            assert_array_equal(data, a)
            # close the mem-mapped file
            del data
            if IS_PYPY:
                break_cycles()
                break_cycles()

    @pytest.mark.xfail(IS_WASM, reason="memmap doesn't work correctly")
    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_save_load_memmap_readwrite(self, filename_type):
        with temppath(suffix='.npy') as path:
            path = filename_type(path)
            a = np.array([[1, 2], [3, 4]], int)
            np.save(path, a)
            b = np.load(path, mmap_mode='r+')
            a[0][0] = 5
            b[0][0] = 5
            del b  # closes the file
            if IS_PYPY:
                break_cycles()
                break_cycles()
            data = np.load(path)
            assert_array_equal(data, a)

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_savez_load(self, filename_type):
        with temppath(suffix='.npz') as path:
            path = filename_type(path)
            np.savez(path, lab='place holder')
            with np.load(path) as data:
                assert_array_equal(data['lab'], 'place holder')

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_savez_compressed_load(self, filename_type):
        with temppath(suffix='.npz') as path:
            path = filename_type(path)
            np.savez_compressed(path, lab='place holder')
            data = np.load(path)
            assert_array_equal(data['lab'], 'place holder')
            data.close()

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_genfromtxt(self, filename_type):
        with temppath(suffix='.txt') as path:
            path = filename_type(path)
            a = np.array([(1, 2), (3, 4)])
            np.savetxt(path, a)
            data = np.genfromtxt(path)
            assert_array_equal(a, data)

    @pytest.mark.parametrize("filename_type", [Path, str])
    @pytest.mark.filterwarnings("ignore:.*recfromtxt.*:DeprecationWarning")
    def test_recfromtxt(self, filename_type):
        with temppath(suffix='.txt') as path:
            path = filename_type(path)
            with open(path, 'w') as f:
                f.write('A,B\n0,1\n2,3')

            kwargs = {"delimiter": ",", "missing_values": "N/A", "names": True}
            test = recfromtxt(path, **kwargs)
            control = np.array([(0, 1), (2, 3)],
                               dtype=[('A', int), ('B', int)])
            assert_(isinstance(test, np.recarray))
            assert_equal(test, control)

    @pytest.mark.parametrize("filename_type", [Path, str])
    @pytest.mark.filterwarnings("ignore:.*recfromcsv.*:DeprecationWarning")
    def test_recfromcsv(self, filename_type):
        with temppath(suffix='.txt') as path:
            path = filename_type(path)
            with open(path, 'w') as f:
                f.write('A,B\n0,1\n2,3')

            kwargs = {
                "missing_values": "N/A", "names": True, "case_sensitive": True
            }
            test = recfromcsv(path, dtype=None, **kwargs)
            control = np.array([(0, 1), (2, 3)],
                               dtype=[('A', int), ('B', int)])
            assert_(isinstance(test, np.recarray))
            assert_equal(test, control)


def test_gzip_load():
    a = np.random.random((5, 5))

    s = BytesIO()
    f = gzip.GzipFile(fileobj=s, mode="w")

    np.save(f, a)
    f.close()
    s.seek(0)

    f = gzip.GzipFile(fileobj=s, mode="r")
    assert_array_equal(np.load(f), a)


# These next two classes encode the minimal API needed to save()/load() arrays.
# The `test_ducktyping` ensures they work correctly
class JustWriter:
    def __init__(self, base):
        self.base = base

    def write(self, s):
        return self.base.write(s)

    def flush(self):
        return self.base.flush()

class JustReader:
    def __init__(self, base):
        self.base = base

    def read(self, n):
        return self.base.read(n)

    def seek(self, off, whence=0):
        return self.base.seek(off, whence)


def test_ducktyping():
    a = np.random.random((5, 5))

    s = BytesIO()
    f = JustWriter(s)

    np.save(f, a)
    f.flush()
    s.seek(0)

    f = JustReader(s)
    assert_array_equal(np.load(f), a)


def test_gzip_loadtxt():
    # Thanks to another windows brokenness, we can't use
    # NamedTemporaryFile: a file created from this function cannot be
    # reopened by another open call. So we first put the gzipped string
    # of the test reference array, write it to a securely opened file,
    # which is then read from by the loadtxt function
    s = BytesIO()
    g = gzip.GzipFile(fileobj=s, mode='w')
    g.write(b'1 2 3\n')
    g.close()

    s.seek(0)
    with temppath(suffix='.gz') as name:
        with open(name, 'wb') as f:
            f.write(s.read())
        res = np.loadtxt(name)
    s.close()

    assert_array_equal(res, [1, 2, 3])


def test_gzip_loadtxt_from_string():
    s = BytesIO()
    f = gzip.GzipFile(fileobj=s, mode="w")
    f.write(b'1 2 3\n')
    f.close()
    s.seek(0)

    f = gzip.GzipFile(fileobj=s, mode="r")
    assert_array_equal(np.loadtxt(f), [1, 2, 3])


def test_npzfile_dict():
    s = BytesIO()
    x = np.zeros((3, 3))
    y = np.zeros((3, 3))

    np.savez(s, x=x, y=y)
    s.seek(0)

    z = np.load(s)

    assert_('x' in z)
    assert_('y' in z)
    assert_('x' in z.keys())
    assert_('y' in z.keys())

    for f, a in z.items():
        assert_(f in ['x', 'y'])
        assert_equal(a.shape, (3, 3))

    for a in z.values():
        assert_equal(a.shape, (3, 3))

    assert_(len(z.items()) == 2)

    for f in z:
        assert_(f in ['x', 'y'])

    assert_('x' in z.keys())
    assert (z.get('x') == z['x']).all()


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_load_refcount():
    # Check that objects returned by np.load are directly freed based on
    # their refcount, rather than needing the gc to collect them.

    f = BytesIO()
    np.savez(f, [1, 2, 3])
    f.seek(0)

    with assert_no_gc_cycles():
        np.load(f)

    f.seek(0)
    dt = [("a", 'u1', 2), ("b", 'u1', 2)]
    with assert_no_gc_cycles():
        x = np.loadtxt(TextIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))


def test_load_multiple_arrays_until_eof():
    f = BytesIO()
    np.save(f, 1)
    np.save(f, 2)
    f.seek(0)
    out1 = np.load(f)
    assert out1 == 1
    out2 = np.load(f)
    assert out2 == 2
    with pytest.raises(EOFError):
        np.load(f)


def test_savez_nopickle():
    obj_array = np.array([1, 'hello'], dtype=object)
    with temppath(suffix='.npz') as tmp:
        np.savez(tmp, obj_array)

    with temppath(suffix='.npz') as tmp:
        with pytest.raises(ValueError, match="Object arrays cannot be saved when.*"):
            np.savez(tmp, obj_array, allow_pickle=False)

    with temppath(suffix='.npz') as tmp:
        np.savez_compressed(tmp, obj_array)

    with temppath(suffix='.npz') as tmp:
        with pytest.raises(ValueError, match="Object arrays cannot be saved when.*"):
            np.savez_compressed(tmp, obj_array, allow_pickle=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_npyio_impl.py ===
"""
IO related functions.
"""
import contextlib
import functools
import itertools
import operator
import os
import pickle
import re
import warnings
import weakref
from collections.abc import Mapping
from operator import itemgetter

import numpy as np
from numpy._core import overrides
from numpy._core._multiarray_umath import _load_from_filelike
from numpy._core.multiarray import packbits, unpackbits
from numpy._core.overrides import finalize_array_function_like, set_module
from numpy._utils import asbytes, asunicode

from . import format
from ._datasource import DataSource  # noqa: F401
from ._format_impl import _MAX_HEADER_SIZE
from ._iotools import (
    ConversionWarning,
    ConverterError,
    ConverterLockError,
    LineSplitter,
    NameValidator,
    StringConverter,
    _decode_line,
    _is_string_like,
    easy_dtype,
    flatten_dtype,
    has_nested_fields,
)

__all__ = [
    'savetxt', 'loadtxt', 'genfromtxt', 'load', 'save', 'savez',
    'savez_compressed', 'packbits', 'unpackbits', 'fromregex'
    ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


class BagObj:
    """
    BagObj(obj)

    Convert attribute look-ups to getitems on the object passed in.

    Parameters
    ----------
    obj : class instance
        Object on which attribute look-up is performed.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib._npyio_impl import BagObj as BO
    >>> class BagDemo:
    ...     def __getitem__(self, key): # An instance of BagObj(BagDemo)
    ...                                 # will call this method when any
    ...                                 # attribute look-up is required
    ...         result = "Doesn't matter what you want, "
    ...         return result + "you're gonna get this"
    ...
    >>> demo_obj = BagDemo()
    >>> bagobj = BO(demo_obj)
    >>> bagobj.hello_there
    "Doesn't matter what you want, you're gonna get this"
    >>> bagobj.I_can_be_anything
    "Doesn't matter what you want, you're gonna get this"

    """

    def __init__(self, obj):
        # Use weakref to make NpzFile objects collectable by refcount
        self._obj = weakref.proxy(obj)

    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        """
        Enables dir(bagobj) to list the files in an NpzFile.

        This also enables tab-completion in an interpreter or IPython.
        """
        return list(object.__getattribute__(self, '_obj').keys())


def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    if not hasattr(file, 'read'):
        file = os.fspath(file)
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, *args, **kwargs)


@set_module('numpy.lib.npyio')
class NpzFile(Mapping):
    """
    NpzFile(fid)

    A dictionary-like object with lazy-loading of files in the zipped
    archive provided on construction.

    `NpzFile` is used to load files in the NumPy ``.npz`` data archive
    format. It assumes that files in the archive have a ``.npy`` extension,
    other files are ignored.

    The arrays and file strings are lazily loaded on either
    getitem access using ``obj['key']`` or attribute lookup using
    ``obj.f.key``. A list of all files (without ``.npy`` extensions) can
    be obtained with ``obj.files`` and the ZipFile object itself using
    ``obj.zip``.

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ``.npy`` extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    f : BagObj instance
        An object on which attribute can be performed as an alternative
        to getitem access on the `NpzFile` instance itself.
    allow_pickle : bool, optional
        Allow loading pickled data. Default: False
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass on to pickle.load.
        These are only useful when loading object arrays saved on
        Python 2.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Parameters
    ----------
    fid : file, str, or pathlib.Path
        The zipped archive to open. This is either a file-like object
        or a string containing the path to the archive.
    own_fid : bool, optional
        Whether NpzFile should close the file handle.
        Requires that `fid` is a file-like object.

    Examples
    --------
    >>> import numpy as np
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)

    >>> npz = np.load(outfile)
    >>> isinstance(npz, np.lib.npyio.NpzFile)
    True
    >>> npz
    NpzFile 'object' with keys: x, y
    >>> sorted(npz.files)
    ['x', 'y']
    >>> npz['x']  # getitem access
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> npz.f.x  # attribute lookup
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    # Make __exit__ safe if zipfile_factory raises an exception
    zip = None
    fid = None
    _MAX_REPR_ARRAY_COUNT = 5

    def __init__(self, fid, own_fid=False, allow_pickle=False,
                 pickle_kwargs=None, *,
                 max_header_size=_MAX_HEADER_SIZE):
        # Import is postponed to here since zipfile depends on gzip, an
        # optional component of the so-called standard library.
        _zip = zipfile_factory(fid)
        _files = _zip.namelist()
        self.files = [name.removesuffix(".npy") for name in _files]
        self._files = dict(zip(self.files, _files))
        self._files.update(zip(_files, _files))
        self.allow_pickle = allow_pickle
        self.max_header_size = max_header_size
        self.pickle_kwargs = pickle_kwargs
        self.zip = _zip
        self.f = BagObj(self)
        if own_fid:
            self.fid = fid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close the file.

        """
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        self.f = None  # break reference cycle

    def __del__(self):
        self.close()

    # Implement the Mapping ABC
    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        try:
            key = self._files[key]
        except KeyError:
            raise KeyError(f"{key} is not a file in the archive") from None
        else:
            with self.zip.open(key) as bytes:
                magic = bytes.read(len(format.MAGIC_PREFIX))
                bytes.seek(0)
                if magic == format.MAGIC_PREFIX:
                    # FIXME: This seems like it will copy strings around
                    #   more than is strictly necessary.  The zipfile
                    #   will read the string and then
                    #   the format.read_array will copy the string
                    #   to another place in memory.
                    #   It would be better if the zipfile could read
                    #   (or at least uncompress) the data
                    #   directly into the array memory.
                    return format.read_array(
                        bytes,
                        allow_pickle=self.allow_pickle,
                        pickle_kwargs=self.pickle_kwargs,
                        max_header_size=self.max_header_size
                    )
                else:
                    return bytes.read(key)

    def __contains__(self, key):
        return (key in self._files)

    def __repr__(self):
        # Get filename or default to `object`
        if isinstance(self.fid, str):
            filename = self.fid
        else:
            filename = getattr(self.fid, "name", "object")

        # Get the name of arrays
        array_names = ', '.join(self.files[:self._MAX_REPR_ARRAY_COUNT])
        if len(self.files) > self._MAX_REPR_ARRAY_COUNT:
            array_names += "..."
        return f"NpzFile {filename!r} with keys: {array_names}"

    # Work around problems with the docstrings in the Mapping methods
    # They contain a `->`, which confuses the type annotation interpretations
    # of sphinx-docs. See gh-25964

    def get(self, key, default=None, /):
        """
        D.get(k,[,d]) returns D[k] if k in D, else d.  d defaults to None.
        """
        return Mapping.get(self, key, default)

    def items(self):
        """
        D.items() returns a set-like object providing a view on the items
        """
        return Mapping.items(self)

    def keys(self):
        """
        D.keys() returns a set-like object providing a view on the keys
        """
        return Mapping.keys(self)

    def values(self):
        """
        D.values() returns a set-like object providing a view on the values
        """
        return Mapping.values(self)


@set_module('numpy')
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding='ASCII', *, max_header_size=_MAX_HEADER_SIZE):
    """
    Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    .. warning:: Loading files that contain object arrays uses the ``pickle``
                 module, which is not secure against erroneous or maliciously
                 constructed data. Consider passing ``allow_pickle=False`` to
                 load data that is known not to contain object arrays for the
                 safer handling of untrusted sources.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        ``seek()`` and ``read()`` methods and must always
        be opened in binary mode.  Pickled files require that the
        file-like object support the ``readline()`` method as well.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode (see
        `numpy.memmap` for a detailed description of the modes).  A
        memory-mapped array is kept on disk. However, it can be accessed
        and sliced like any ndarray.  Memory mapping is especially useful
        for accessing small fragments of large files without reading the
        entire file into memory.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons for
        disallowing pickles include security, as loading pickled data can
        execute arbitrary code. If pickles are disallowed, loading object
        arrays will fail. Default: False
    fix_imports : bool, optional
        Only useful when loading Python 2 generated pickled files,
        which includes npy/npz files containing object arrays. If `fix_imports`
        is True, pickle will try to map the old Python 2 names to the new names
        used in Python 3.
    encoding : str, optional
        What encoding to use when reading Python 2 strings. Only useful when
        loading Python 2 generated pickled files, which includes
        npy/npz files containing object arrays. Values other than 'latin1',
        'ASCII', and 'bytes' are not allowed, as they can corrupt numerical
        data. Default: 'ASCII'
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance
        of NpzFile class must be closed to avoid leaking file descriptors.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.
    UnpicklingError
        If ``allow_pickle=True``, but the file cannot be loaded as a pickle.
    ValueError
        The file contains an object array, but ``allow_pickle=False`` given.
    EOFError
        When calling ``np.load`` multiple times on the same file handle,
        if all data has already been read

    See Also
    --------
    save, savez, savez_compressed, loadtxt
    memmap : Create a memory-map to an array stored in a file on disk.
    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

    Notes
    -----
    - If the file contains pickle data, then whatever object is stored
      in the pickle is returned.
    - If the file is a ``.npy`` file, then a single array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is
      returned, containing ``{filename: array}`` key-value pairs, one for
      each file in the archive.
    - If the file is a ``.npz`` file, the returned value supports the
      context manager protocol in a similar fashion to the open function::

        with load('foo.npz') as data:
            a = data['a']

      The underlying file descriptor is closed when exiting the 'with'
      block.

    Examples
    --------
    >>> import numpy as np

    Store data to disk, and load it again:

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> np.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Store compressed data to disk, and load it again:

    >>> a=np.array([[1, 2, 3], [4, 5, 6]])
    >>> b=np.array([1, 2])
    >>> np.savez('/tmp/123.npz', a=a, b=b)
    >>> data = np.load('/tmp/123.npz')
    >>> data['a']
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> data['b']
    array([1, 2])
    >>> data.close()

    Mem-map the stored array, and then access the second row
    directly from disk:

    >>> X = np.load('/tmp/123.npy', mmap_mode='r')
    >>> X[1, :]
    memmap([4, 5, 6])

    """
    if encoding not in ('ASCII', 'latin1', 'bytes'):
        # The 'encoding' value for pickle also affects what encoding
        # the serialized binary data of NumPy arrays is loaded
        # in. Pickle does not pass on the encoding information to
        # NumPy. The unpickling code in numpy._core.multiarray is
        # written to assume that unicode data appearing where binary
        # should be is in 'latin1'. 'bytes' is also safe, as is 'ASCII'.
        #
        # Other encoding values can corrupt binary data, and we
        # purposefully disallow them. For the same reason, the errors=
        # argument is not exposed, as values other than 'strict'
        # result can similarly silently corrupt numerical data.
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")

    pickle_kwargs = {'encoding': encoding, 'fix_imports': fix_imports}

    with contextlib.ExitStack() as stack:
        if hasattr(file, 'read'):
            fid = file
            own_fid = False
        else:
            fid = stack.enter_context(open(os.fspath(file), "rb"))
            own_fid = True

        # Code to distinguish from NumPy binary files and pickles.
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06'  # empty zip files start with this
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        if not magic:
            raise EOFError("No data left in file")
        # If the file size is less than N, we need to make sure not
        # to seek past the beginning of the file
        fid.seek(-min(N, len(magic)), 1)  # back-up
        if magic.startswith((_ZIP_PREFIX, _ZIP_SUFFIX)):
            # zip-file (assume .npz)
            # Potentially transfer file ownership to NpzFile
            stack.pop_all()
            ret = NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle,
                          pickle_kwargs=pickle_kwargs,
                          max_header_size=max_header_size)
            return ret
        elif magic == format.MAGIC_PREFIX:
            # .npy file
            if mmap_mode:
                if allow_pickle:
                    max_header_size = 2**64
                return format.open_memmap(file, mode=mmap_mode,
                                          max_header_size=max_header_size)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle,
                                         pickle_kwargs=pickle_kwargs,
                                         max_header_size=max_header_size)
        else:
            # Try a pickle
            if not allow_pickle:
                raise ValueError(
                    "This file contains pickled (object) data. If you trust "
                    "the file you can load it unsafely using the "
                    "`allow_pickle=` keyword argument or `pickle.load()`.")
            try:
                return pickle.load(fid, **pickle_kwargs)
            except Exception as e:
                raise pickle.UnpicklingError(
                    f"Failed to interpret file {file!r} as a pickle") from e


def _save_dispatcher(file, arr, allow_pickle=None, fix_imports=None):
    return (arr,)


@array_function_dispatch(_save_dispatcher)
def save(file, arr, allow_pickle=True, fix_imports=np._NoValue):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        File or filename to which the data is saved. If file is a file-object,
        then the filename is unchanged.  If file is a string or Path,
        a ``.npy`` extension will be appended to the filename if it does not
        already have one.
    arr : array_like
        Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    fix_imports : bool, optional
        The `fix_imports` flag is deprecated and has no effect.

        .. deprecated:: 2.1
            This flag is ignored since NumPy 1.17 and was only needed to
            support loading in Python 2 some files written in Python 3.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` archive
    savetxt, load

    Notes
    -----
    For a description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    Any data saved to the file is appended to the end of the file.

    Examples
    --------
    >>> import numpy as np

    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


    >>> with open('test.npy', 'wb') as f:
    ...     np.save(f, np.array([1, 2]))
    ...     np.save(f, np.array([1, 3]))
    >>> with open('test.npy', 'rb') as f:
    ...     a = np.load(f)
    ...     b = np.load(f)
    >>> print(a, b)
    # [1 2] [1 3]
    """
    if fix_imports is not np._NoValue:
        # Deprecated 2024-05-16, NumPy 2.1
        warnings.warn(
            "The 'fix_imports' flag is deprecated and has no effect. "
            "(Deprecated in NumPy 2.1)",
            DeprecationWarning, stacklevel=2)
    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        file_ctx = open(file, "wb")

    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle,
                           pickle_kwargs={'fix_imports': fix_imports})


def _savez_dispatcher(file, *args, allow_pickle=True, **kwds):
    yield from args
    yield from kwds.values()


@array_function_dispatch(_savez_dispatcher)
def savez(file, *args, allow_pickle=True, **kwds):
    """Save several arrays into a single file in uncompressed ``.npz`` format.

    Provide arrays as keyword arguments to store them under the
    corresponding name in the output file: ``savez(fn, x=x, y=y)``.

    If arrays are specified as positional arguments, i.e., ``savez(fn,
    x, y)``, their names will be `arr_0`, `arr_1`, etc.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    args : Arguments, optional
        Arrays to save to the file. Please use keyword arguments (see
        `kwds` below) to assign names to arrays.  Arrays specified as
        args will be named "arr_0", "arr_1", and so on.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    kwds : Keyword arguments, optional
        Arrays to save to the file. Each array will be saved to the
        output file with its corresponding keyword name.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.
    savetxt : Save an array to a file as plain text.
    savez_compressed : Save several arrays into a compressed ``.npz`` archive

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is not compressed and each file
    in the archive contains one variable in ``.npy`` format. For a
    description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    When opening the saved ``.npz`` file with `load` a `~lib.npyio.NpzFile`
    object is returned. This is a dictionary-like object which can be queried
    for its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Keys passed in `kwds` are used as filenames inside the ZIP archive.
    Therefore, keys should be valid filenames; e.g., avoid keys that begin with
    ``/`` or contain ``.``.

    When naming variables with keyword arguments, it is not possible to name a
    variable ``file``, as this would cause the ``file`` argument to be defined
    twice in the call to ``savez``.

    Examples
    --------
    >>> import numpy as np
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)

    Using `savez` with \\*args, the arrays are saved with default names.

    >>> np.savez(outfile, x, y)
    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['arr_0', 'arr_1']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using `savez` with \\**kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)
    >>> npzfile = np.load(outfile)
    >>> sorted(npzfile.files)
    ['x', 'y']
    >>> npzfile['x']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    _savez(file, args, kwds, False, allow_pickle=allow_pickle)


def _savez_compressed_dispatcher(file, *args, allow_pickle=True, **kwds):
    yield from args
    yield from kwds.values()


@array_function_dispatch(_savez_compressed_dispatcher)
def savez_compressed(file, *args, allow_pickle=True, **kwds):
    """
    Save several arrays into a single file in compressed ``.npz`` format.

    Provide arrays as keyword arguments to store them under the
    corresponding name in the output file: ``savez_compressed(fn, x=x, y=y)``.

    If arrays are specified as positional arguments, i.e.,
    ``savez_compressed(fn, x, y)``, their names will be `arr_0`, `arr_1`, etc.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    args : Arguments, optional
        Arrays to save to the file. Please use keyword arguments (see
        `kwds` below) to assign names to arrays.  Arrays specified as
        args will be named "arr_0", "arr_1", and so on.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    kwds : Keyword arguments, optional
        Arrays to save to the file. Each array will be saved to the
        output file with its corresponding keyword name.

    Returns
    -------
    None

    See Also
    --------
    numpy.save : Save a single array to a binary file in NumPy format.
    numpy.savetxt : Save an array to a file as plain text.
    numpy.savez : Save several arrays into an uncompressed ``.npz`` file format
    numpy.load : Load the files created by savez_compressed.

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is compressed with
    ``zipfile.ZIP_DEFLATED`` and each file in the archive contains one variable
    in ``.npy`` format. For a description of the ``.npy`` format, see
    :py:mod:`numpy.lib.format`.


    When opening the saved ``.npz`` file with `load` a `~lib.npyio.NpzFile`
    object is returned. This is a dictionary-like object which can be queried
    for its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Examples
    --------
    >>> import numpy as np
    >>> test_array = np.random.rand(3, 2)
    >>> test_vector = np.random.rand(4)
    >>> np.savez_compressed('/tmp/123', a=test_array, b=test_vector)
    >>> loaded = np.load('/tmp/123.npz')
    >>> print(np.array_equal(test_array, loaded['a']))
    True
    >>> print(np.array_equal(test_vector, loaded['b']))
    True

    """
    _savez(file, args, kwds, True, allow_pickle=allow_pickle)


def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile

    if not hasattr(file, 'write'):
        file = os.fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError(
                f"Cannot use un-named variables and keyword {key}")
        namedict[key] = val

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zipf = zipfile_factory(file, mode="w", compression=compression)
    try:
        for key, val in namedict.items():
            fname = key + '.npy'
            val = np.asanyarray(val)
            # always force zip64, gh-10776
            with zipf.open(fname, 'w', force_zip64=True) as fid:
                format.write_array(fid, val,
                                   allow_pickle=allow_pickle,
                                   pickle_kwargs=pickle_kwargs)
    finally:
        zipf.close()


def _ensure_ndmin_ndarray_check_param(ndmin):
    """Just checks if the param ndmin is supported on
        _ensure_ndmin_ndarray. It is intended to be used as
        verification before running anything expensive.
        e.g. loadtxt, genfromtxt
    """
    # Check correctness of the values of `ndmin`
    if ndmin not in [0, 1, 2]:
        raise ValueError(f"Illegal value of ndmin keyword: {ndmin}")

def _ensure_ndmin_ndarray(a, *, ndmin: int):
    """This is a helper function of loadtxt and genfromtxt to ensure
        proper minimum dimension as requested

        ndim : int. Supported values 1, 2, 3
                    ^^ whenever this changes, keep in sync with
                       _ensure_ndmin_ndarray_check_param
    """
    # Verify that the array has at least dimensions `ndmin`.
    # Tweak the size and shape of the arrays - remove extraneous dimensions
    if a.ndim > ndmin:
        a = np.squeeze(a)
    # and ensure we have the minimum number of dimensions asked for
    # - has to be in this order for the odd case ndmin=1, a.squeeze().ndim=0
    if a.ndim < ndmin:
        if ndmin == 1:
            a = np.atleast_1d(a)
        elif ndmin == 2:
            a = np.atleast_2d(a).T

    return a


# amount of lines loadtxt reads in one chunk, can be overridden for testing
_loadtxt_chunksize = 50000


def _check_nonneg_int(value, name="argument"):
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _preprocess_comments(iterable, comments, encoding):
    """
    Generator that consumes a line iterated iterable and strips out the
    multiple (or multi-character) comments from lines.
    This is a pre-processing step to achieve feature parity with loadtxt
    (we assume that this feature is a nieche feature).
    """
    for line in iterable:
        if isinstance(line, bytes):
            # Need to handle conversion here, or the splitting would fail
            line = line.decode(encoding)

        for c in comments:
            line = line.split(c, 1)[0]

        yield line


# The number of rows we read in one go if confronted with a parametric dtype
_loadtxt_chunksize = 50000


def _read(fname, *, delimiter=',', comment='#', quote='"',
          imaginary_unit='j', usecols=None, skiplines=0,
          max_rows=None, converters=None, ndmin=None, unpack=False,
          dtype=np.float64, encoding=None):
    r"""
    Read a NumPy array from a text file.
    This is a helper function for loadtxt.

    Parameters
    ----------
    fname : file, str, or pathlib.Path
        The filename or the file to be read.
    delimiter : str, optional
        Field delimiter of the fields in line of the file.
        Default is a comma, ','.  If None any sequence of whitespace is
        considered a delimiter.
    comment : str or sequence of str or None, optional
        Character that begins a comment.  All text from the comment
        character to the end of the line is ignored.
        Multiple comments or multiple-character comment strings are supported,
        but may be slower and `quote` must be empty if used.
        Use None to disable all use of comments.
    quote : str or None, optional
        Character that is used to quote string fields. Default is '"'
        (a double quote). Use None to disable quote support.
    imaginary_unit : str, optional
        Character that represent the imaginary unit `sqrt(-1)`.
        Default is 'j'.
    usecols : array_like, optional
        A one-dimensional array of integer column numbers.  These are the
        columns from the file to be included in the array.  If this value
        is not given, all the columns are used.
    skiplines : int, optional
        Number of lines to skip before interpreting the data in the file.
    max_rows : int, optional
        Maximum number of rows of data to read.  Default is to read the
        entire file.
    converters : dict or callable, optional
        A function to parse all columns strings into the desired value, or
        a dictionary mapping column number to a parser function.
        E.g. if column 0 is a date string: ``converters = {0: datestr2num}``.
        Converters can also be used to provide a default value for missing
        data, e.g. ``converters = lambda s: float(s.strip() or 0)`` will
        convert empty fields to 0.
        Default: None
    ndmin : int, optional
        Minimum dimension of the array returned.
        Allowed values are 0, 1 or 2.  Default is 0.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = read(...)``.  When used with a structured
        data-type, arrays are returned for each field.  Default is False.
    dtype : numpy data type
        A NumPy dtype instance, can be a structured dtype to map to the
        columns of the file.
    encoding : str, optional
        Encoding used to decode the inputfile. The special value 'bytes'
        (the default) enables backwards-compatible behavior for `converters`,
        ensuring that inputs to the converter functions are encoded
        bytes objects. The special value 'bytes' has no additional effect if
        ``converters=None``. If encoding is ``'bytes'`` or ``None``, the
        default system encoding is used.

    Returns
    -------
    ndarray
        NumPy array.
    """
    # Handle special 'bytes' keyword for encoding
    byte_converters = False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True

    if dtype is None:
        raise TypeError("a dtype must be provided.")
    dtype = np.dtype(dtype)

    read_dtype_via_object_chunks = None
    if dtype.kind in 'SUM' and dtype in {
            np.dtype("S0"), np.dtype("U0"), np.dtype("M8"), np.dtype("m8")}:
        # This is a legacy "flexible" dtype.  We do not truly support
        # parametric dtypes currently (no dtype discovery step in the core),
        # but have to support these for backward compatibility.
        read_dtype_via_object_chunks = dtype
        dtype = np.dtype(object)

    if usecols is not None:
        # Allow usecols to be a single int or a sequence of ints, the C-code
        # handles the rest
        try:
            usecols = list(usecols)
        except TypeError:
            usecols = [usecols]

    _ensure_ndmin_ndarray_check_param(ndmin)

    if comment is None:
        comments = None
    else:
        # assume comments are a sequence of strings
        if "" in comment:
            raise ValueError(
                "comments cannot be an empty string. Use comments=None to "
                "disable comments."
            )
        comments = tuple(comment)
        comment = None
        if len(comments) == 0:
            comments = None  # No comments at all
        elif len(comments) == 1:
            # If there is only one comment, and that comment has one character,
            # the normal parsing can deal with it just fine.
            if isinstance(comments[0], str) and len(comments[0]) == 1:
                comment = comments[0]
                comments = None
        # Input validation if there are multiple comment characters
        elif delimiter in comments:
            raise TypeError(
                f"Comment characters '{comments}' cannot include the "
                f"delimiter '{delimiter}'"
            )

    # comment is now either a 1 or 0 character string or a tuple:
    if comments is not None:
        # Note: An earlier version support two character comments (and could
        #       have been extended to multiple characters, we assume this is
        #       rare enough to not optimize for.
        if quote is not None:
            raise ValueError(
                "when multiple comments or a multi-character comment is "
                "given, quotes are not supported.  In this case quotechar "
                "must be set to None.")

    if len(imaginary_unit) != 1:
        raise ValueError('len(imaginary_unit) must be 1.')

    _check_nonneg_int(skiplines)
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    else:
        # Passing -1 to the C code means "read the entire file".
        max_rows = -1

    fh_closing_ctx = contextlib.nullcontext()
    filelike = False
    try:
        if isinstance(fname, os.PathLike):
            fname = os.fspath(fname)
        if isinstance(fname, str):
            fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            if encoding is None:
                encoding = getattr(fh, 'encoding', 'latin1')

            fh_closing_ctx = contextlib.closing(fh)
            data = fh
            filelike = True
        else:
            if encoding is None:
                encoding = getattr(fname, 'encoding', 'latin1')
            data = iter(fname)
    except TypeError as e:
        raise ValueError(
            f"fname must be a string, filehandle, list of strings,\n"
            f"or generator. Got {type(fname)} instead.") from e

    with fh_closing_ctx:
        if comments is not None:
            if filelike:
                data = iter(data)
                filelike = False
            data = _preprocess_comments(data, comments, encoding)

        if read_dtype_via_object_chunks is None:
            arr = _load_from_filelike(
                data, delimiter=delimiter, comment=comment, quote=quote,
                imaginary_unit=imaginary_unit,
                usecols=usecols, skiplines=skiplines, max_rows=max_rows,
                converters=converters, dtype=dtype,
                encoding=encoding, filelike=filelike,
                byte_converters=byte_converters)

        else:
            # This branch reads the file into chunks of object arrays and then
            # casts them to the desired actual dtype.  This ensures correct
            # string-length and datetime-unit discovery (like `arr.astype()`).
            # Due to chunking, certain error reports are less clear, currently.
            if filelike:
                data = iter(data)  # cannot chunk when reading from file
                filelike = False

            c_byte_converters = False
            if read_dtype_via_object_chunks == "S":
                c_byte_converters = True  # Use latin1 rather than ascii

            chunks = []
            while max_rows != 0:
                if max_rows < 0:
                    chunk_size = _loadtxt_chunksize
                else:
                    chunk_size = min(_loadtxt_chunksize, max_rows)

                next_arr = _load_from_filelike(
                    data, delimiter=delimiter, comment=comment, quote=quote,
                    imaginary_unit=imaginary_unit,
                    usecols=usecols, skiplines=skiplines, max_rows=chunk_size,
                    converters=converters, dtype=dtype,
                    encoding=encoding, filelike=filelike,
                    byte_converters=byte_converters,
                    c_byte_converters=c_byte_converters)
                # Cast here already.  We hope that this is better even for
                # large files because the storage is more compact.  It could
                # be adapted (in principle the concatenate could cast).
                chunks.append(next_arr.astype(read_dtype_via_object_chunks))

                skiplines = 0  # Only have to skip for first chunk
                if max_rows >= 0:
                    max_rows -= chunk_size
                if len(next_arr) < chunk_size:
                    # There was less data than requested, so we are done.
                    break

            # Need at least one chunk, but if empty, the last one may have
            # the wrong shape.
            if len(chunks) > 1 and len(chunks[-1]) == 0:
                del chunks[-1]
            if len(chunks) == 1:
                arr = chunks[0]
            else:
                arr = np.concatenate(chunks, axis=0)

    # NOTE: ndmin works as advertised for structured dtypes, but normally
    #       these would return a 1D result plus the structured dimension,
    #       so ndmin=2 adds a third dimension even when no squeezing occurs.
    #       A `squeeze=False` could be a better solution (pandas uses squeeze).
    arr = _ensure_ndmin_ndarray(arr, ndmin=ndmin)

    if arr.shape:
        if arr.shape[0] == 0:
            warnings.warn(
                f'loadtxt: input contained no data: "{fname}"',
                category=UserWarning,
                stacklevel=3
            )

    if unpack:
        # Unpack structured dtypes if requested:
        dt = arr.dtype
        if dt.names is not None:
            # For structured arrays, return an array for each field.
            return [arr[field] for field in dt.names]
        else:
            return arr.T
    else:
        return arr


@finalize_array_function_like
@set_module('numpy')
def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding=None, max_rows=None, *, quotechar=None,
            like=None):
    r"""
    Load data from a text file.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read.  If the filename
        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
        that generators must return bytes or strings. The strings
        in a list or produced by a generator are treated as lines.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str or None, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The character used to separate the values. For backwards compatibility,
        byte strings will be decoded as 'latin1'. The default is whitespace.

        .. versionchanged:: 1.23.0
           Only single character delimiters are supported. Newline characters
           cannot be used as the delimiter.

    converters : dict or callable, optional
        Converter functions to customize value parsing. If `converters` is
        callable, the function is applied to all columns, else it must be a
        dict that maps column number to a parser function.
        See examples for further details.
        Default: None.

        .. versionchanged:: 1.23.0
           The ability to pass a single callable to be applied to all columns
           was added.

    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The default, None, results in all columns being read.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``.  When used with a
        structured data-type, arrays are returned for each field.
        Default is False.
    ndmin : int, optional
        The returned array will have at least `ndmin` dimensions.
        Otherwise mono-dimensional axes will be squeezed.
        Legal values: 0 (default), 1 or 2.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams.
        The special value 'bytes' enables backward compatibility workarounds
        that ensures you receive byte arrays as results if possible and passes
        'latin1' encoded strings to converters. Override this value to receive
        unicode arrays and pass strings as input to converters.  If set to None
        the system default is used. The default value is None.

        .. versionchanged:: 2.0
            Before NumPy 2, the default was ``'bytes'`` for Python 2
            compatibility. The default is now ``None``.

    max_rows : int, optional
        Read `max_rows` rows of content after `skiprows` lines. The default is
        to read all the rows. Note that empty rows containing no data such as
        empty lines and comment lines are not counted towards `max_rows`,
        while such lines are counted in `skiprows`.

        .. versionchanged:: 1.23.0
            Lines containing no data, including comment lines (e.g., lines
            starting with '#' or as specified via `comments`) are not counted
            towards `max_rows`.
    quotechar : unicode character or None, optional
        The character used to denote the start and end of a quoted item.
        Occurrences of the delimiter or comment characters are ignored within
        a quoted item. The default value is ``quotechar=None``, which means
        quoting support is disabled.

        If two consecutive instances of `quotechar` are found within a quoted
        field, the first is treated as an escape character. See examples.

        .. versionadded:: 1.23.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    load, fromstring, fromregex
    genfromtxt : Load data with missing values handled as specified.
    scipy.io.loadmat : reads MATLAB data files

    Notes
    -----
    This function aims to be a fast reader for simply formatted files.  The
    `genfromtxt` function provides more sophisticated handling of, e.g.,
    lines with missing values.

    Each row in the input text file must have the same number of values to be
    able to read all values. If all rows do not have same number of values, a
    subset of up to n columns (where n is the least number of values present
    in all rows) can be read by specifying the columns via `usecols`.

    The strings produced by the Python float.hex method can be used as
    input for floats.

    Examples
    --------
    >>> import numpy as np
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\n2 3")
    >>> np.loadtxt(c)
    array([[0., 1.],
           [2., 3.]])

    >>> d = StringIO("M 21 72\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([(b'M', 21, 72.), (b'F', 35, 58.)],
          dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\n3,0,4")
    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([1., 3.])
    >>> y
    array([2., 4.])

    The `converters` argument is used to specify functions to preprocess the
    text prior to parsing. `converters` can be a dictionary that maps
    preprocessing functions to each column:

    >>> s = StringIO("1.618, 2.296\n3.141, 4.669\n")
    >>> conv = {
    ...     0: lambda x: np.floor(float(x)),  # conversion fn for column 0
    ...     1: lambda x: np.ceil(float(x)),  # conversion fn for column 1
    ... }
    >>> np.loadtxt(s, delimiter=",", converters=conv)
    array([[1., 3.],
           [3., 5.]])

    `converters` can be a callable instead of a dictionary, in which case it
    is applied to all columns:

    >>> s = StringIO("0xDE 0xAD\n0xC0 0xDE")
    >>> import functools
    >>> conv = functools.partial(int, base=16)
    >>> np.loadtxt(s, converters=conv)
    array([[222., 173.],
           [192., 222.]])

    This example shows how `converters` can be used to convert a field
    with a trailing minus sign into a negative number.

    >>> s = StringIO("10.01 31.25-\n19.22 64.31\n17.57- 63.94")
    >>> def conv(fld):
    ...     return -float(fld[:-1]) if fld.endswith("-") else float(fld)
    ...
    >>> np.loadtxt(s, converters=conv)
    array([[ 10.01, -31.25],
           [ 19.22,  64.31],
           [-17.57,  63.94]])

    Using a callable as the converter can be particularly useful for handling
    values with different formatting, e.g. floats with underscores:

    >>> s = StringIO("1 2.7 100_000")
    >>> np.loadtxt(s, converters=float)
    array([1.e+00, 2.7e+00, 1.e+05])

    This idea can be extended to automatically handle values specified in
    many different formats, such as hex values:

    >>> def conv(val):
    ...     try:
    ...         return float(val)
    ...     except ValueError:
    ...         return float.fromhex(val)
    >>> s = StringIO("1, 2.5, 3_000, 0b4, 0x1.4000000000000p+2")
    >>> np.loadtxt(s, delimiter=",", converters=conv)
    array([1.0e+00, 2.5e+00, 3.0e+03, 1.8e+02, 5.0e+00])

    Or a format where the ``-`` sign comes after the number:

    >>> s = StringIO("10.01 31.25-\n19.22 64.31\n17.57- 63.94")
    >>> conv = lambda x: -float(x[:-1]) if x.endswith("-") else float(x)
    >>> np.loadtxt(s, converters=conv)
    array([[ 10.01, -31.25],
           [ 19.22,  64.31],
           [-17.57,  63.94]])

    Support for quoted fields is enabled with the `quotechar` parameter.
    Comment and delimiter characters are ignored when they appear within a
    quoted item delineated by `quotechar`:

    >>> s = StringIO('"alpha, #42", 10.0\n"beta, #64", 2.0\n')
    >>> dtype = np.dtype([("label", "U12"), ("value", float)])
    >>> np.loadtxt(s, dtype=dtype, delimiter=",", quotechar='"')
    array([('alpha, #42', 10.), ('beta, #64',  2.)],
          dtype=[('label', '<U12'), ('value', '<f8')])

    Quoted fields can be separated by multiple whitespace characters:

    >>> s = StringIO('"alpha, #42"       10.0\n"beta, #64" 2.0\n')
    >>> dtype = np.dtype([("label", "U12"), ("value", float)])
    >>> np.loadtxt(s, dtype=dtype, delimiter=None, quotechar='"')
    array([('alpha, #42', 10.), ('beta, #64',  2.)],
          dtype=[('label', '<U12'), ('value', '<f8')])

    Two consecutive quote characters within a quoted field are treated as a
    single escaped character:

    >>> s = StringIO('"Hello, my name is ""Monty""!"')
    >>> np.loadtxt(s, dtype="U", delimiter=",", quotechar='"')
    array('Hello, my name is "Monty"!', dtype='<U26')

    Read subset of columns when all rows do not contain equal number of values:

    >>> d = StringIO("1 2\n2 4\n3 9 12\n4 16 20")
    >>> np.loadtxt(d, usecols=(0, 1))
    array([[ 1.,  2.],
           [ 2.,  4.],
           [ 3.,  9.],
           [ 4., 16.]])

    """

    if like is not None:
        return _loadtxt_with_like(
            like, fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows, usecols=usecols,
            unpack=unpack, ndmin=ndmin, encoding=encoding,
            max_rows=max_rows
        )

    if isinstance(delimiter, bytes):
        delimiter.decode("latin1")

    if dtype is None:
        dtype = np.float64

    comment = comments
    # Control character type conversions for Py3 convenience
    if comment is not None:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [
            x.decode('latin1') if isinstance(x, bytes) else x for x in comment]
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode('latin1')

    arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
                converters=converters, skiplines=skiprows, usecols=usecols,
                unpack=unpack, ndmin=ndmin, encoding=encoding,
                max_rows=max_rows, quote=quotechar)

    return arr


_loadtxt_with_like = array_function_dispatch()(loadtxt)


def _savetxt_dispatcher(fname, X, fmt=None, delimiter=None, newline=None,
                        header=None, footer=None, comments=None,
                        encoding=None):
    return (X,)


@array_function_dispatch(_savetxt_dispatcher)
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : filename, file handle or pathlib.Path
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : 1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:

        * a single specifier, ``fmt='%.4e'``, resulting in numbers formatted
          like ``' (%s+%sj)' % (fmt, fmt)``
        * a full string specifying every real and imaginary part, e.g.
          ``' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'`` for 3 columns
        * a list of specifiers, one per column - in this case, the real
          and imaginary part must have separate specifiers,
          e.g. ``['%.3e + %.3ej', '(%.15e%+.15ej)']`` for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
    header : str, optional
        String that will be written at the beginning of the file.
    footer : str, optional
        String that will be written at the end of the file.
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ',  as expected by e.g.
        ``numpy.loadtxt``.
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output
        streams. If the encoding is something other than 'bytes' or 'latin1'
        you will not be able to load the file in NumPy versions < 1.14. Default
        is 'latin1'.

    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into an uncompressed ``.npz`` archive
    savez_compressed : Save several arrays into a compressed ``.npz`` archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to precede result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <https://docs.python.org/library/string.html#format-specification-mini-language>`_,
           Python Documentation.

    Examples
    --------
    >>> import numpy as np
    >>> x = y = z = np.arange(0.0,5.0,1.0)
    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    """

    class WriteWrap:
        """Convert to bytes on bytestream inputs.

        """
        def __init__(self, fh, encoding):
            self.fh = fh
            self.encoding = encoding
            self.do_write = self.first_write

        def close(self):
            self.fh.close()

        def write(self, v):
            self.do_write(v)

        def write_bytes(self, v):
            if isinstance(v, bytes):
                self.fh.write(v)
            else:
                self.fh.write(v.encode(self.encoding))

        def write_normal(self, v):
            self.fh.write(asunicode(v))

        def first_write(self, v):
            try:
                self.write_normal(v)
                self.write = self.write_normal
            except TypeError:
                # input is probably a bytestream
                self.write_bytes(v)
                self.write = self.write_bytes

    own_fh = False
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if _is_string_like(fname):
        # datasource doesn't support creating a new file ...
        open(fname, 'wt').close()
        fh = np.lib._datasource.open(fname, 'wt', encoding=encoding)
        own_fh = True
    elif hasattr(fname, 'write'):
        # wrap to handle byte output streams
        fh = WriteWrap(fname, encoding or 'latin1')
    else:
        raise ValueError('fname must be a string or file handle')

    try:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        if X.ndim == 0 or X.ndim > 2:
            raise ValueError(
                "Expected 1D or 2D array, got %dD array instead" % X.ndim)
        elif X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(X.dtype.names)
        else:
            ncol = X.shape[1]

        iscomplex_X = np.iscomplexobj(X)
        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError(f'fmt has wrong shape.  {str(fmt)}')
            format = delimiter.join(fmt)
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError(f'fmt has wrong number of % formats:  {fmt}')
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [f' ({fmt}+{fmt}j)', ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError(f'invalid fmt: {fmt!r}')

        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(comments + header + newline)
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                    row2.extend((number.real, number.imag))
                s = format % tuple(row2) + newline
                fh.write(s.replace('+-', '-'))
        else:
            for row in X:
                try:
                    v = format % tuple(row) + newline
                except TypeError as e:
                    raise TypeError("Mismatch between array dtype ('%s') and "
                                    "format specifier ('%s')"
                                    % (str(X.dtype), format)) from e
                fh.write(v)

        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(comments + footer + newline)
    finally:
        if own_fh:
            fh.close()


@set_module('numpy')
def fromregex(file, regexp, dtype, encoding=None):
    r"""
    Construct an array from a text file, using regular expression parsing.

    The returned array is always a structured array, and is constructed from
    all matches of the regular expression in the file. Groups in the regular
    expression are converted to fields of the structured array.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Filename or file object to read.

        .. versionchanged:: 1.22.0
            Now accepts `os.PathLike` implementations.

    regexp : str or regexp
        Regular expression used to parse the file.
        Groups in the regular expression correspond to fields in the dtype.
    dtype : dtype or list of dtypes
        Dtype for the structured array; must be a structured datatype.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams.

    Returns
    -------
    output : ndarray
        The output array, containing the part of the content of `file` that
        was matched by `regexp`. `output` is always a structured array.

    Raises
    ------
    TypeError
        When `dtype` is not a valid dtype for a structured array.

    See Also
    --------
    fromstring, loadtxt

    Notes
    -----
    Dtypes for structured arrays can be specified in several forms, but all
    forms specify at least the data type and field name. For details see
    `basics.rec`.

    Examples
    --------
    >>> import numpy as np
    >>> from io import StringIO
    >>> text = StringIO("1312 foo\n1534  bar\n444   qux")

    >>> regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]
    >>> output = np.fromregex(text, regexp,
    ...                       [('num', np.int64), ('key', 'S3')])
    >>> output
    array([(1312, b'foo'), (1534, b'bar'), ( 444, b'qux')],
          dtype=[('num', '<i8'), ('key', 'S3')])
    >>> output['num']
    array([1312, 1534,  444])

    """
    own_fh = False
    if not hasattr(file, "read"):
        file = os.fspath(file)
        file = np.lib._datasource.open(file, 'rt', encoding=encoding)
        own_fh = True

    try:
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        if dtype.names is None:
            raise TypeError('dtype must be a structured datatype.')

        content = file.read()
        if isinstance(content, bytes) and isinstance(regexp, str):
            regexp = asbytes(regexp)

        if not hasattr(regexp, 'match'):
            regexp = re.compile(regexp)
        seq = regexp.findall(content)
        if seq and not isinstance(seq[0], tuple):
            # Only one group is in the regexp.
            # Create the new array as a single data-type and then
            #   re-interpret as a single-field structured array.
            newdtype = np.dtype(dtype[dtype.names[0]])
            output = np.array(seq, dtype=newdtype)
            output.dtype = dtype
        else:
            output = np.array(seq, dtype=dtype)

        return output
    finally:
        if own_fh:
            file.close()


#####--------------------------------------------------------------------------
#---- --- ASCII functions ---
#####--------------------------------------------------------------------------


@finalize_array_function_like
@set_module('numpy')
def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None,
               deletechars=''.join(sorted(NameValidator.defaultdeletechars)),  # noqa: B008
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt="f%i", unpack=None, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding=None,
               *, ndmin=0, like=None):
    """
    Load data from a text file, with missing values handled as specified.

    Each line past the first `skip_header` lines is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read.  If the filename
        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
        that generators must return bytes or strings. The strings
        in a list or produced by a generator are treated as lines.
    dtype : dtype, optional
        Data type of the resulting array.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : str, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded.
    delimiter : str, int, or sequence, optional
        The string used to separate values.  By default, any consecutive
        whitespaces act as delimiter.  An integer or sequence of integers
        can also be provided as width(s) of each field.
    skiprows : int, optional
        `skiprows` was removed in numpy 1.10. Please use `skip_header` instead.
    skip_header : int, optional
        The number of lines to skip at the beginning of the file.
    skip_footer : int, optional
        The number of lines to skip at the end of the file.
    converters : variable, optional
        The set of functions that convert the data of a column to a value.
        The converters can also be used to provide a default value
        for missing data: ``converters = {3: lambda s: float(s or 0)}``.
    missing : variable, optional
        `missing` was removed in numpy 1.10. Please use `missing_values`
        instead.
    missing_values : variable, optional
        The set of strings corresponding to missing data.
    filling_values : variable, optional
        The set of values to be used as default when the data are missing.
    usecols : sequence, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, str, sequence}, optional
        If `names` is True, the field names are read from the first line after
        the first `skip_header` lines. This line can optionally be preceded
        by a comment delimiter. Any content before the comment delimiter is
        discarded. If `names` is a sequence or a single-string of
        comma-separated names, the names will be used to define the field
        names in a structured dtype. If `names` is None, the names of the
        dtype fields will be used, if any.
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended with an
        underscore: for example, `file` would become `file_`.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    defaultfmt : str, optional
        A format used to define default field names, such as "f%i" or "f_%02i".
    autostrip : bool, optional
        Whether to automatically strip white spaces from the variables.
    replace_space : char, optional
        Character(s) used in replacement of white spaces in the variable
        names. By default, use a '_'.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        If True, field names are case sensitive.
        If False or 'upper', field names are converted to upper case.
        If 'lower', field names are converted to lower case.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = genfromtxt(...)``.  When used with a
        structured data-type, arrays are returned for each field.
        Default is False.
    usemask : bool, optional
        If True, return a masked array.
        If False, return a regular array.
    loose : bool, optional
        If True, do not raise errors for invalid values.
    invalid_raise : bool, optional
        If True, an exception is raised if an inconsistency is detected in the
        number of columns.
        If False, a warning is emitted and the offending lines are skipped.
    max_rows : int,  optional
        The maximum number of rows to read. Must not be used with skip_footer
        at the same time.  If given, the value must be at least 1. Default is
        to read the entire file.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply when `fname`
        is a file object. The special value 'bytes' enables backward
        compatibility workarounds that ensure that you receive byte arrays
        when possible and passes latin1 encoded strings to converters.
        Override this value to receive unicode arrays and pass strings
        as input to converters.  If set to None the system default is used.
        The default value is 'bytes'.

        .. versionchanged:: 2.0
            Before NumPy 2, the default was ``'bytes'`` for Python 2
            compatibility. The default is now ``None``.

    ndmin : int, optional
        Same parameter as `loadtxt`

        .. versionadded:: 1.23.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Data read from the text file. If `usemask` is True, this is a
        masked array.

    See Also
    --------
    numpy.loadtxt : equivalent function when no data is missing.

    Notes
    -----
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When variables are named (either by a flexible dtype or with a `names`
      sequence), there must not be any header in the file (else a ValueError
      exception is raised).
    * Individual values are not stripped of spaces by default.
      When using a custom converter, make sure the function does remove spaces.
    * Custom converters may receive unexpected values due to dtype
      discovery.

    References
    ----------
    .. [1] NumPy User Guide, section `I/O with NumPy
           <https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_.

    Examples
    --------
    >>> from io import StringIO
    >>> import numpy as np

    Comma delimited file with mixed dtype

    >>> s = StringIO("1,1.3,abcde")
    >>> data = np.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
    ... ('mystring','S5')], delimiter=",")
    >>> data
    array((1, 1.3, b'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

    Using dtype = None

    >>> _ = s.seek(0) # needed for StringIO example only
    >>> data = np.genfromtxt(s, dtype=None,
    ... names = ['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '<U5')])

    Specifying dtype and names

    >>> _ = s.seek(0)
    >>> data = np.genfromtxt(s, dtype="i8,f8,S5",
    ... names=['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, b'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

    An example with fixed-width columns

    >>> s = StringIO("11.3abcde")
    >>> data = np.genfromtxt(s, dtype=None, names=['intvar','fltvar','strvar'],
    ...     delimiter=[1,3,5])
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', '<U5')])

    An example to show comments

    >>> f = StringIO('''
    ... text,# of chars
    ... hello world,11
    ... numpy,5''')
    >>> np.genfromtxt(f, dtype='S12,S12', delimiter=',')
    array([(b'text', b''), (b'hello world', b'11'), (b'numpy', b'5')],
      dtype=[('f0', 'S12'), ('f1', 'S12')])

    """

    if like is not None:
        return _genfromtxt_with_like(
            like, fname, dtype=dtype, comments=comments, delimiter=delimiter,
            skip_header=skip_header, skip_footer=skip_footer,
            converters=converters, missing_values=missing_values,
            filling_values=filling_values, usecols=usecols, names=names,
            excludelist=excludelist, deletechars=deletechars,
            replace_space=replace_space, autostrip=autostrip,
            case_sensitive=case_sensitive, defaultfmt=defaultfmt,
            unpack=unpack, usemask=usemask, loose=loose,
            invalid_raise=invalid_raise, max_rows=max_rows, encoding=encoding,
            ndmin=ndmin,
        )

    _ensure_ndmin_ndarray_check_param(ndmin)

    if max_rows is not None:
        if skip_footer:
            raise ValueError(
                    "The keywords 'skip_footer' and 'max_rows' can not be "
                    "specified at the same time.")
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")

    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr
    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        raise TypeError(
            "The input argument 'converter' should be a valid dictionary "
            "(got '%s' instead)" % type(user_converters))

    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    else:
        byte_converters = False

    # Initialize the filehandle, the LineSplitter and the NameValidator
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if isinstance(fname, str):
        fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
        fid_ctx = contextlib.closing(fid)
    else:
        fid = fname
        fid_ctx = contextlib.nullcontext(fid)
    try:
        fhd = iter(fid)
    except TypeError as e:
        raise TypeError(
            "fname must be a string, a filehandle, a sequence of strings,\n"
            f"or an iterator of strings. Got {type(fname)} instead."
        ) from e
    with fid_ctx:
        split_line = LineSplitter(delimiter=delimiter, comments=comments,
                                  autostrip=autostrip, encoding=encoding)
        validate_names = NameValidator(excludelist=excludelist,
                                       deletechars=deletechars,
                                       case_sensitive=case_sensitive,
                                       replace_space=replace_space)

        # Skip the first `skip_header` rows
        try:
            for i in range(skip_header):
                next(fhd)

            # Keep on until we find the first valid values
            first_values = None

            while not first_values:
                first_line = _decode_line(next(fhd), encoding)
                if (names is True) and (comments is not None):
                    if comments in first_line:
                        first_line = (
                            ''.join(first_line.split(comments)[1:]))
                first_values = split_line(first_line)
        except StopIteration:
            # return an empty array if the datafile is empty
            first_line = ''
            first_values = []
            warnings.warn(
                f'genfromtxt: Empty input file: "{fname}"', stacklevel=2
            )

        # Should we take the first values as names ?
        if names is True:
            fval = first_values[0].strip()
            if comments is not None:
                if fval in comments:
                    del first_values[0]

        # Check the columns to use: make sure `usecols` is a list
        if usecols is not None:
            try:
                usecols = [_.strip() for _ in usecols.split(",")]
            except AttributeError:
                try:
                    usecols = list(usecols)
                except TypeError:
                    usecols = [usecols, ]
        nbcols = len(usecols or first_values)

        # Check the names and overwrite the dtype.names if needed
        if names is True:
            names = validate_names([str(_.strip()) for _ in first_values])
            first_line = ''
        elif _is_string_like(names):
            names = validate_names([_.strip() for _ in names.split(',')])
        elif names:
            names = validate_names(names)
        # Get the dtype
        if dtype is not None:
            dtype = easy_dtype(dtype, defaultfmt=defaultfmt, names=names,
                               excludelist=excludelist,
                               deletechars=deletechars,
                               case_sensitive=case_sensitive,
                               replace_space=replace_space)
        # Make sure the names is a list (for 2.5)
        if names is not None:
            names = list(names)

        if usecols:
            for (i, current) in enumerate(usecols):
                # if usecols is a list of names, convert to a list of indices
                if _is_string_like(current):
                    usecols[i] = names.index(current)
                elif current < 0:
                    usecols[i] = current + len(first_values)
            # If the dtype is not None, make sure we update it
            if (dtype is not None) and (len(dtype) > nbcols):
                descr = dtype.descr
                dtype = np.dtype([descr[_] for _ in usecols])
                names = list(dtype.names)
            # If `names` is not None, update the names
            elif (names is not None) and (len(names) > nbcols):
                names = [names[_] for _ in usecols]
        elif (names is not None) and (dtype is not None):
            names = list(dtype.names)

        # Process the missing values ...............................
        # Rename missing_values for convenience
        user_missing_values = missing_values or ()
        if isinstance(user_missing_values, bytes):
            user_missing_values = user_missing_values.decode('latin1')

        # Define the list of missing_values (one column: one list)
        missing_values = [[''] for _ in range(nbcols)]

        # We have a dictionary: process it field by field
        if isinstance(user_missing_values, dict):
            # Loop on the items
            for (key, val) in user_missing_values.items():
                # Is the key a string ?
                if _is_string_like(key):
                    try:
                        # Transform it into an integer
                        key = names.index(key)
                    except ValueError:
                        # We couldn't find it: the name must have been dropped
                        continue
                # Redefine the key as needed if it's a column number
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                # Transform the value as a list of string
                if isinstance(val, (list, tuple)):
                    val = [str(_) for _ in val]
                else:
                    val = [str(val), ]
                # Add the value(s) to the current list of missing
                if key is None:
                    # None acts as default
                    for miss in missing_values:
                        miss.extend(val)
                else:
                    missing_values[key].extend(val)
        # We have a sequence : each item matches a column
        elif isinstance(user_missing_values, (list, tuple)):
            for (value, entry) in zip(user_missing_values, missing_values):
                value = str(value)
                if value not in entry:
                    entry.append(value)
        # We have a string : apply it to all entries
        elif isinstance(user_missing_values, str):
            user_value = user_missing_values.split(",")
            for entry in missing_values:
                entry.extend(user_value)
        # We have something else: apply it to all entries
        else:
            for entry in missing_values:
                entry.extend([str(user_missing_values)])

        # Process the filling_values ...............................
        # Rename the input for convenience
        user_filling_values = filling_values
        if user_filling_values is None:
            user_filling_values = []
        # Define the default
        filling_values = [None] * nbcols
        # We have a dictionary : update each entry individually
        if isinstance(user_filling_values, dict):
            for (key, val) in user_filling_values.items():
                if _is_string_like(key):
                    try:
                        # Transform it into an integer
                        key = names.index(key)
                    except ValueError:
                        # We couldn't find it: the name must have been dropped
                        continue
                # Redefine the key if it's a column number
                # and usecols is defined
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                # Add the value to the list
                filling_values[key] = val
        # We have a sequence : update on a one-to-one basis
        elif isinstance(user_filling_values, (list, tuple)):
            n = len(user_filling_values)
            if (n <= nbcols):
                filling_values[:n] = user_filling_values
            else:
                filling_values = user_filling_values[:nbcols]
        # We have something else : use it for all entries
        else:
            filling_values = [user_filling_values] * nbcols

        # Initialize the converters ................................
        if dtype is None:
            # Note: we can't use a [...]*nbcols, as we would have 3 times
            # the same converter, instead of 3 different converters.
            converters = [
                StringConverter(None, missing_values=miss, default=fill)
                for (miss, fill) in zip(missing_values, filling_values)
            ]
        else:
            dtype_flat = flatten_dtype(dtype, flatten_base=True)
            # Initialize the converters
            if len(dtype_flat) > 1:
                # Flexible type : get a converter from each dtype
                zipit = zip(dtype_flat, missing_values, filling_values)
                converters = [StringConverter(dt,
                                              locked=True,
                                              missing_values=miss,
                                              default=fill)
                              for (dt, miss, fill) in zipit]
            else:
                # Set to a default converter (but w/ different missing values)
                zipit = zip(missing_values, filling_values)
                converters = [StringConverter(dtype,
                                              locked=True,
                                              missing_values=miss,
                                              default=fill)
                              for (miss, fill) in zipit]
        # Update the converters to use the user-defined ones
        uc_update = []
        for (j, conv) in user_converters.items():
            # If the converter is specified by column names,
            # use the index instead
            if _is_string_like(j):
                try:
                    j = names.index(j)
                    i = j
                except ValueError:
                    continue
            elif usecols:
                try:
                    i = usecols.index(j)
                except ValueError:
                    # Unused converter specified
                    continue
            else:
                i = j
            # Find the value to test - first_line is not filtered by usecols:
            if len(first_line):
                testing_value = first_values[j]
            else:
                testing_value = None
            if conv is bytes:
                user_conv = asbytes
            elif byte_converters:
                # Converters may use decode to workaround numpy's old
                # behavior, so encode the string again before passing
                # to the user converter.
                def tobytes_first(x, conv):
                    if type(x) is bytes:
                        return conv(x)
                    return conv(x.encode("latin1"))
                user_conv = functools.partial(tobytes_first, conv=conv)
            else:
                user_conv = conv
            converters[i].update(user_conv, locked=True,
                                 testing_value=testing_value,
                                 default=filling_values[i],
                                 missing_values=missing_values[i],)
            uc_update.append((i, user_conv))
        # Make sure we have the corrected keys in user_converters...
        user_converters.update(uc_update)

        # Fixme: possible error as following variable never used.
        # miss_chars = [_.missing_values for _ in converters]

        # Initialize the output lists ...
        # ... rows
        rows = []
        append_to_rows = rows.append
        # ... masks
        if usemask:
            masks = []
            append_to_masks = masks.append
        # ... invalid
        invalid = []
        append_to_invalid = invalid.append

        # Parse each line
        for (i, line) in enumerate(itertools.chain([first_line, ], fhd)):
            values = split_line(line)
            nbvalues = len(values)
            # Skip an empty line
            if nbvalues == 0:
                continue
            if usecols:
                # Select only the columns we need
                try:
                    values = [values[_] for _ in usecols]
                except IndexError:
                    append_to_invalid((i + skip_header + 1, nbvalues))
                    continue
            elif nbvalues != nbcols:
                append_to_invalid((i + skip_header + 1, nbvalues))
                continue
            # Store the values
            append_to_rows(tuple(values))
            if usemask:
                append_to_masks(tuple(v.strip() in m
                                       for (v, m) in zip(values,
                                                         missing_values)))
            if len(rows) == max_rows:
                break

    # Upgrade the converters (if needed)
    if dtype is None:
        for (i, converter) in enumerate(converters):
            current_column = [itemgetter(i)(_m) for _m in rows]
            try:
                converter.iterupgrade(current_column)
            except ConverterLockError:
                errmsg = f"Converter #{i} is locked and cannot be upgraded: "
                current_column = map(itemgetter(i), rows)
                for (j, value) in enumerate(current_column):
                    try:
                        converter.upgrade(value)
                    except (ConverterError, ValueError):
                        line_number = j + 1 + skip_header
                        errmsg += f"(occurred line #{line_number} for value '{value}')"
                        raise ConverterError(errmsg)

    # Check that we don't have invalid values
    nbinvalid = len(invalid)
    if nbinvalid > 0:
        nbrows = len(rows) + nbinvalid - skip_footer
        # Construct the error message
        template = f"    Line #%i (got %i columns instead of {nbcols})"
        if skip_footer > 0:
            nbinvalid_skipped = len([_ for _ in invalid
                                     if _[0] > nbrows + skip_header])
            invalid = invalid[:nbinvalid - nbinvalid_skipped]
            skip_footer -= nbinvalid_skipped
#
#            nbrows -= skip_footer
#            errmsg = [template % (i, nb)
#                      for (i, nb) in invalid if i < nbrows]
#        else:
        errmsg = [template % (i, nb)
                  for (i, nb) in invalid]
        if len(errmsg):
            errmsg.insert(0, "Some errors were detected !")
            errmsg = "\n".join(errmsg)
            # Raise an exception ?
            if invalid_raise:
                raise ValueError(errmsg)
            # Issue a warning ?
            else:
                warnings.warn(errmsg, ConversionWarning, stacklevel=2)

    # Strip the last skip_footer data
    if skip_footer > 0:
        rows = rows[:-skip_footer]
        if usemask:
            masks = masks[:-skip_footer]

    # Convert each value according to the converter:
    # We want to modify the list in place to avoid creating a new one...
    if loose:
        rows = list(
            zip(*[[conv._loose_call(_r) for _r in map(itemgetter(i), rows)]
                  for (i, conv) in enumerate(converters)]))
    else:
        rows = list(
            zip(*[[conv._strict_call(_r) for _r in map(itemgetter(i), rows)]
                  for (i, conv) in enumerate(converters)]))

    # Reset the dtype
    data = rows
    if dtype is None:
        # Get the dtypes from the types of the converters
        column_types = [conv.type for conv in converters]
        # Find the columns with strings...
        strcolidx = [i for (i, v) in enumerate(column_types)
                     if v == np.str_]

        if byte_converters and strcolidx:
            # convert strings back to bytes for backward compatibility
            warnings.warn(
                "Reading unicode strings without specifying the encoding "
                "argument is deprecated. Set the encoding, use None for the "
                "system default.",
                np.exceptions.VisibleDeprecationWarning, stacklevel=2)

            def encode_unicode_cols(row_tup):
                row = list(row_tup)
                for i in strcolidx:
                    row[i] = row[i].encode('latin1')
                return tuple(row)

            try:
                data = [encode_unicode_cols(r) for r in data]
            except UnicodeEncodeError:
                pass
            else:
                for i in strcolidx:
                    column_types[i] = np.bytes_

        # Update string types to be the right length
        sized_column_types = column_types.copy()
        for i, col_type in enumerate(column_types):
            if np.issubdtype(col_type, np.character):
                n_chars = max(len(row[i]) for row in data)
                sized_column_types[i] = (col_type, n_chars)

        if names is None:
            # If the dtype is uniform (before sizing strings)
            base = {
                c_type
                for c, c_type in zip(converters, column_types)
                if c._checked}
            if len(base) == 1:
                uniform_type, = base
                (ddtype, mdtype) = (uniform_type, bool)
            else:
                ddtype = [(defaultfmt % i, dt)
                          for (i, dt) in enumerate(sized_column_types)]
                if usemask:
                    mdtype = [(defaultfmt % i, bool)
                              for (i, dt) in enumerate(sized_column_types)]
        else:
            ddtype = list(zip(names, sized_column_types))
            mdtype = list(zip(names, [bool] * len(sized_column_types)))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        # Overwrite the initial dtype names if needed
        if names and dtype.names is not None:
            dtype.names = names
        # Case 1. We have a structured type
        if len(dtype_flat) > 1:
            # Nested dtype, eg [('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # First, create the array using a flattened dtype:
            # [('a', int), ('b1', int), ('b2', float)]
            # Then, view the array using the specified dtype.
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    raise NotImplementedError(
                        "Nested fields involving objects are not supported...")
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            # Now, process the rowmasks the same way
            if usemask:
                rowmasks = np.array(
                    masks, dtype=np.dtype([('', bool) for t in dtype_flat]))
                # Construct the new dtype
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        # Case #2. We have a basic dtype
        else:
            # We used some user-defined converters
            if user_converters:
                ishomogeneous = True
                descr = []
                for i, ttype in enumerate([conv.type for conv in converters]):
                    # Keep the dtype of the current converter
                    if i in user_converters:
                        ishomogeneous &= (ttype == dtype.type)
                        if np.issubdtype(ttype, np.character):
                            ttype = (ttype, max(len(row[i]) for row in data))
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                # So we changed the dtype ?
                if not ishomogeneous:
                    # We have more than one field
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    # We have only one field: drop the name if not needed.
                    else:
                        dtype = np.dtype(ttype)
            #
            output = np.array(data, dtype)
            if usemask:
                if dtype.names is not None:
                    mdtype = [(_, bool) for _ in dtype.names]
                else:
                    mdtype = bool
                outputmask = np.array(masks, dtype=mdtype)
    # Try to take care of the missing data we missed
    names = output.dtype.names
    if usemask and names:
        for (name, conv) in zip(names, converters):
            missing_values = [conv(_) for _ in conv.missing_values
                              if _ != '']
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    # Construct the final array
    if usemask:
        output = output.view(MaskedArray)
        output._mask = outputmask

    output = _ensure_ndmin_ndarray(output, ndmin=ndmin)

    if unpack:
        if names is None:
            return output.T
        elif len(names) == 1:
            # squeeze single-name dtypes too
            return output[names[0]]
        else:
            # For structured arrays with multiple fields,
            # return an array for each field.
            return [output[field] for field in names]
    return output


_genfromtxt_with_like = array_function_dispatch()(genfromtxt)


def recfromtxt(fname, **kwargs):
    """
    Load ASCII data from a file and return it in a record array.

    If ``usemask=False`` a standard `recarray` is returned,
    if ``usemask=True`` a MaskedRecords array is returned.

    .. deprecated:: 2.0
        Use `numpy.genfromtxt` instead.

    Parameters
    ----------
    fname, kwargs : For a description of input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function

    Notes
    -----
    By default, `dtype` is None, which means that the data-type of the output
    array will be determined from the data.

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`recfromtxt` is deprecated, "
        "use `numpy.genfromtxt` instead."
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    kwargs.setdefault("dtype", None)
    usemask = kwargs.get('usemask', False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


def recfromcsv(fname, **kwargs):
    """
    Load ASCII data stored in a comma-separated file.

    The returned array is a record array (if ``usemask=False``, see
    `recarray`) or a masked record array (if ``usemask=True``,
    see `ma.mrecords.MaskedRecords`).

    .. deprecated:: 2.0
        Use `numpy.genfromtxt` with comma as `delimiter` instead.

    Parameters
    ----------
    fname, kwargs : For a description of input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function to load ASCII data.

    Notes
    -----
    By default, `dtype` is None, which means that the data-type of the output
    array will be determined from the data.

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`recfromcsv` is deprecated, "
        "use `numpy.genfromtxt` with comma as `delimiter` instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    # Set default kwargs for genfromtxt as relevant to csv import.
    kwargs.setdefault("case_sensitive", "lower")
    kwargs.setdefault("names", True)
    kwargs.setdefault("delimiter", ",")
    kwargs.setdefault("dtype", None)
    output = genfromtxt(fname, **kwargs)

    usemask = kwargs.get("usemask", False)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output

# === atelier-kyo-manager/app\utils\ai_image_crawler.py ===
# image_crawler.py ─ DeepLab v3 (MobileNetV3-ONNX 版)
# 依存: torch torchvision onnx opencv-python numpy icrawler requests
# ------------------------------------------------------------
from __future__ import annotations
import cv2, numpy as np, requests
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from pathlib import Path
from typing import List

# 1. モデルファイル
MODEL_DIR  = Path(__file__).parent / "models"
ONNX_FILE  = MODEL_DIR / "deeplabv3_mnv3.onnx"

# 2. ダウンロード処理は廃止（ローカルに無ければ明示エラー）
def ensure_model() -> None:
    if ONNX_FILE.exists():
        print("✓ DeepLab ONNX モデルは既に存在します")
        return
    raise FileNotFoundError(
        f"{ONNX_FILE.name} がありません。\n"
        "1) app/utils/models で make_deeplab_onnx.py を実行して生成する\n"
        "2) 生成した deeplabv3_mnv3.onnx を models フォルダに置いて再実行してください"
    )

# 3. 深度学習セグメンテーション
class DeepLabSeg:
    def __init__(self, onnx: Path, use_cuda=False):
        self.net = cv2.dnn.readNetFromONNX(str(onnx))
        backend = cv2.dnn.DNN_BACKEND_CUDA if use_cuda else cv2.dnn.DNN_BACKEND_OPENCV
        target  = cv2.dnn.DNN_TARGET_CUDA  if use_cuda else cv2.dnn.DNN_TARGET_CPU
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)

    def remove_bg(self, img: np.ndarray, blur=5) -> np.ndarray:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/127.5, (513, 513), (127.5,)*3, swapRB=True)
        self.net.setInput(blob)
        mask = self.net.forward()[0].argmax(0).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        fg_mask = cv2.medianBlur(mask*255, blur) if blur else mask*255
        fg = cv2.bitwise_and(img, img, mask=fg_mask)
        return np.where(fg_mask[..., None] == 255, fg, 255)

# 4. 画像検索＋背景除去
def crawl_and_remove_bg(keyword: str, max_num=30,
                        engines: List[str] | None = None,
                        use_cuda=False) -> Path:
    engines = engines or ["bing", "google"]
    out_dir = Path(__file__).parent / "images" / keyword.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if "bing" in engines:
        BingImageCrawler(storage={"root_dir": str(out_dir)}).crawl(keyword, max_num)
    if "google" in engines:
        GoogleImageCrawler(storage={"root_dir": str(out_dir)}).crawl(keyword, max_num)

    ensure_model()
    seg = DeepLabSeg(ONNX_FILE, use_cuda)

    for p in out_dir.glob("*.jpg"):
        try:
            img = cv2.imread(str(p))
            out_path = p.with_stem(p.stem + "_bg").with_suffix(".png")
            cv2.imwrite(str(out_path), seg.remove_bg(img))
            print("✓", p.name)
        except Exception as e:
            print("×", p.name, e)
    return out_dir

def crawl_images(keyword, max_num=50, engines=None):
    return crawl_and_remove_bg(keyword, max_num, engines or ["bing", "google"])

# 5. CLI
if __name__ == "__main__":
    kw = input("検索キーワードを入力してください：")
    print("完了→", crawl_and_remove_bg(kw, 50, ["bing"]).resolve())

# === atelier-kyo-manager/app\utils\utils\ai_image_crawler.py ===
# image_crawler.py ─ DeepLab v3 (MobileNetV3-ONNX 版)
# 依存: torch torchvision onnx opencv-python numpy icrawler requests
# ------------------------------------------------------------
from __future__ import annotations
import cv2, numpy as np, requests
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from pathlib import Path
from typing import List

# 1. モデルファイル
MODEL_DIR  = Path(__file__).parent / "models"
ONNX_FILE  = MODEL_DIR / "deeplabv3_mnv3.onnx"

# 2. ダウンロード処理は廃止（ローカルに無ければ明示エラー）
def ensure_model() -> None:
    if ONNX_FILE.exists():
        print("✓ DeepLab ONNX モデルは既に存在します")
        return
    raise FileNotFoundError(
        f"{ONNX_FILE.name} がありません。\n"
        "1) app/utils/models で make_deeplab_onnx.py を実行して生成する\n"
        "2) 生成した deeplabv3_mnv3.onnx を models フォルダに置いて再実行してください"
    )

# 3. 深度学習セグメンテーション
class DeepLabSeg:
    def __init__(self, onnx: Path, use_cuda=False):
        self.net = cv2.dnn.readNetFromONNX(str(onnx))
        backend = cv2.dnn.DNN_BACKEND_CUDA if use_cuda else cv2.dnn.DNN_BACKEND_OPENCV
        target  = cv2.dnn.DNN_TARGET_CUDA  if use_cuda else cv2.dnn.DNN_TARGET_CPU
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)

    def remove_bg(self, img: np.ndarray, blur=5) -> np.ndarray:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/127.5, (513, 513), (127.5,)*3, swapRB=True)
        self.net.setInput(blob)
        mask = self.net.forward()[0].argmax(0).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        fg_mask = cv2.medianBlur(mask*255, blur) if blur else mask*255
        fg = cv2.bitwise_and(img, img, mask=fg_mask)
        return np.where(fg_mask[..., None] == 255, fg, 255)

# 4. 画像検索＋背景除去
def crawl_and_remove_bg(keyword: str, max_num=30,
                        engines: List[str] | None = None,
                        use_cuda=False) -> Path:
    engines = engines or ["bing", "google"]
    out_dir = Path(__file__).parent / "images" / keyword.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if "bing" in engines:
        BingImageCrawler(storage={"root_dir": str(out_dir)}).crawl(keyword, max_num)
    if "google" in engines:
        GoogleImageCrawler(storage={"root_dir": str(out_dir)}).crawl(keyword, max_num)

    ensure_model()
    seg = DeepLabSeg(ONNX_FILE, use_cuda)

    for p in out_dir.glob("*.jpg"):
        try:
            img = cv2.imread(str(p))
            out_path = p.with_stem(p.stem + "_bg").with_suffix(".png")
            cv2.imwrite(str(out_path), seg.remove_bg(img))
            print("✓", p.name)
        except Exception as e:
            print("×", p.name, e)
    return out_dir

def crawl_images(keyword, max_num=50, engines=None):
    return crawl_and_remove_bg(keyword, max_num, engines or ["bing", "google"])

# 5. CLI
if __name__ == "__main__":
    kw = input("検索キーワードを入力してください：")
    print("完了→", crawl_and_remove_bg(kw, 50, ["bing"]).resolve())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_nanfunctions_impl.py ===
"""
Functions that ignore NaN.

Functions
---------

- `nanmin` -- minimum non-NaN value
- `nanmax` -- maximum non-NaN value
- `nanargmin` -- index of minimum non-NaN value
- `nanargmax` -- index of maximum non-NaN value
- `nansum` -- sum of non-NaN values
- `nanprod` -- product of non-NaN values
- `nancumsum` -- cumulative sum of non-NaN values
- `nancumprod` -- cumulative product of non-NaN values
- `nanmean` -- mean of non-NaN values
- `nanvar` -- variance of non-NaN values
- `nanstd` -- standard deviation of non-NaN values
- `nanmedian` -- median of non-NaN values
- `nanquantile` -- qth quantile of non-NaN values
- `nanpercentile` -- qth percentile of non-NaN values

"""
import functools
import warnings

import numpy as np
import numpy._core.numeric as _nx
from numpy._core import overrides
from numpy.lib import _function_base_impl as fnb
from numpy.lib._function_base_impl import _weights_are_valid

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
    'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod',
    'nancumsum', 'nancumprod', 'nanquantile'
    ]


def _nan_mask(a, out=None):
    """
    Parameters
    ----------
    a : array-like
        Input array with at least 1 dimension.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output and will prevent the allocation of a new array.

    Returns
    -------
    y : bool ndarray or True
        A bool array where ``np.nan`` positions are marked with ``False``
        and other positions are marked with ``True``. If the type of ``a``
        is such that it can't possibly contain ``np.nan``, returns ``True``.
    """
    # we assume that a is an array for this private function

    if a.dtype.kind not in 'fc':
        return True

    y = np.isnan(a, out=out)
    y = np.invert(y, out=y)
    return y

def _replace_nan(a, val):
    """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Note that scalars will end up as array scalars, which is important
    for using the result as the value of the out argument in some
    operations.

    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.

    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.

    """
    a = np.asanyarray(a)

    if a.dtype == np.object_:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        mask = np.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, np.inexact):
        mask = np.isnan(a)
    else:
        mask = None

    if mask is not None:
        a = np.array(a, subok=True, copy=True)
        np.copyto(a, val, where=mask)

    return a, mask


def _copyto(a, val, mask):
    """
    Replace values in `a` with NaN where `mask` is True.  This differs from
    copyto in that it will deal with the case where `a` is a numpy scalar.

    Parameters
    ----------
    a : ndarray or numpy scalar
        Array or numpy scalar some of whose values are to be replaced
        by val.
    val : numpy scalar
        Value used a replacement.
    mask : ndarray, scalar
        Boolean array. Where True the corresponding element of `a` is
        replaced by `val`. Broadcasts.

    Returns
    -------
    res : ndarray, scalar
        Array with elements replaced or scalar `val`.

    """
    if isinstance(a, np.ndarray):
        np.copyto(a, val, where=mask, casting='unsafe')
    else:
        a = a.dtype.type(val)
    return a


def _remove_nan_1d(arr1d, second_arr1d=None, overwrite_input=False):
    """
    Equivalent to arr1d[~arr1d.isnan()], but in a different order

    Presumably faster as it incurs fewer copies

    Parameters
    ----------
    arr1d : ndarray
        Array to remove nans from
    second_arr1d : ndarray or None
        A second array which will have the same positions removed as arr1d.
    overwrite_input : bool
        True if `arr1d` can be modified in place

    Returns
    -------
    res : ndarray
        Array with nan elements removed
    second_res : ndarray or None
        Second array with nan element positions of first array removed.
    overwrite_input : bool
        True if `res` can be modified in place, given the constraint on the
        input
    """
    if arr1d.dtype == object:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = np.isnan(arr1d)

    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=6)
        if second_arr1d is None:
            return arr1d[:0], None, True
        else:
            return arr1d[:0], second_arr1d[:0], True
    elif s.size == 0:
        return arr1d, second_arr1d, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        # select non-nans at end of array
        enonan = arr1d[-s.size:][~c[-s.size:]]
        # fill nans in beginning of array with non-nans of end
        arr1d[s[:enonan.size]] = enonan

        if second_arr1d is None:
            return arr1d[:-s.size], None, True
        else:
            if not overwrite_input:
                second_arr1d = second_arr1d.copy()
            enonan = second_arr1d[-s.size:][~c[-s.size:]]
            second_arr1d[s[:enonan.size]] = enonan

            return arr1d[:-s.size], second_arr1d[:-s.size], True


def _divide_by_count(a, b, out=None):
    """
    Compute a/b ignoring invalid results. If `a` is an array the division
    is done in place. If `a` is a scalar, then its type is preserved in the
    output. If out is None, then a is used instead so that the division
    is in place. Note that this is only called with `a` an inexact type.

    Parameters
    ----------
    a : {ndarray, numpy scalar}
        Numerator. Expected to be of inexact type but not checked.
    b : {ndarray, numpy scalar}
        Denominator.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.

    Returns
    -------
    ret : {ndarray, numpy scalar}
        The return value is a/b. If `a` was an ndarray the division is done
        in place. If `a` is a numpy scalar, the division preserves its type.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        if isinstance(a, np.ndarray):
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        elif out is None:
            # Precaution against reduced object arrays
            try:
                return a.dtype.type(a / b)
            except AttributeError:
                return a / b
        else:
            # This is questionable, but currently a numpy scalar can
            # be output to a zero dimensional array.
            return np.divide(a, b, out=out, casting='unsafe')


def _nanmin_dispatcher(a, axis=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nanmin_dispatcher)
def nanmin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
           where=np._NoValue):
    """
    Return minimum of an array or minimum along an axis, ignoring any NaNs.
    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
    Nan is returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the minimum is computed. The default is to compute
        the minimum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `min` method
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanmin : ndarray
        An array with the same shape as `a`, with the specified axis
        removed.  If `a` is a 0-d array, or if axis is None, an ndarray
        scalar is returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amax, fmax, maximum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.min.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmin(a)
    1.0
    >>> np.nanmin(a, axis=0)
    array([1.,  2.])
    >>> np.nanmin(a, axis=1)
    array([1.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmin([1, 2, np.nan, np.inf])
    1.0
    >>> np.nanmin([1, 2, np.nan, -np.inf])
    -inf

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if initial is not np._NoValue:
        kwargs['initial'] = initial
    if where is not np._NoValue:
        kwargs['where'] = where

    if (type(a) is np.ndarray or type(a) is np.memmap) and a.dtype != np.object_:
        # Fast, but not safe for subclasses of ndarray, or object arrays,
        # which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=2)
    else:
        # Slow, but safe for subclasses of ndarray
        a, mask = _replace_nan(a, +np.inf)
        res = np.amin(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # Check for all-NaN axis
        kwargs.pop("initial", None)
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=2)
    return res


def _nanmax_dispatcher(a, axis=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nanmax_dispatcher)
def nanmax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
           where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis, ignoring any
    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
    raised and NaN is returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the maximum is computed. The default is to compute
        the maximum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If the value is anything but the default, then
        `keepdims` will be passed through to the `max` method
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanmax : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an ndarray scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amin, fmin, minimum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.max.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmax(a)
    3.0
    >>> np.nanmax(a, axis=0)
    array([3.,  2.])
    >>> np.nanmax(a, axis=1)
    array([2.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmax([1, 2, np.nan, -np.inf])
    2.0
    >>> np.nanmax([1, 2, np.nan, np.inf])
    inf

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if initial is not np._NoValue:
        kwargs['initial'] = initial
    if where is not np._NoValue:
        kwargs['where'] = where

    if (type(a) is np.ndarray or type(a) is np.memmap) and a.dtype != np.object_:
        # Fast, but not safe for subclasses of ndarray, or object arrays,
        # which do not implement isnan (gh-9009), or fmax correctly (gh-8975)
        res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=2)
    else:
        # Slow, but safe for subclasses of ndarray
        a, mask = _replace_nan(a, -np.inf)
        res = np.amax(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # Check for all-NaN axis
        kwargs.pop("initial", None)
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=2)
    return res


def _nanargmin_dispatcher(a, axis=None, out=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_nanargmin_dispatcher)
def nanargmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
    cannot be trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

        .. versionadded:: 1.22.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmin, nanargmax

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmin(a)
    0
    >>> np.nanargmin(a)
    2
    >>> np.nanargmin(a, axis=0)
    array([1, 1])
    >>> np.nanargmin(a, axis=1)
    array([1, 0])

    """
    a, mask = _replace_nan(a, np.inf)
    if mask is not None and mask.size:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    res = np.argmin(a, axis=axis, out=out, keepdims=keepdims)
    return res


def _nanargmax_dispatcher(a, axis=None, out=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_nanargmax_dispatcher)
def nanargmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the
    results cannot be trusted if a slice contains only NaNs and -Infs.


    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

        .. versionadded:: 1.22.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmax, nanargmin

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmax(a)
    0
    >>> np.nanargmax(a)
    1
    >>> np.nanargmax(a, axis=0)
    array([1, 0])
    >>> np.nanargmax(a, axis=1)
    array([1, 1])

    """
    a, mask = _replace_nan(a, -np.inf)
    if mask is not None and mask.size:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    res = np.argmax(a, axis=axis, out=out, keepdims=keepdims)
    return res


def _nansum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nansum_dispatcher)
def nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
           initial=np._NoValue, where=np._NoValue):
    """
    Return the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    In NumPy versions <= 1.9.0 Nan is returned for slices that are all-NaN or
    empty. In later versions zero is returned.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the sum is computed. The default is to compute the
        sum of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nansum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    numpy.sum : Sum across array propagating NaNs.
    isnan : Show which elements are NaN.
    isfinite : Show which elements are not NaN or +/-inf.

    Notes
    -----
    If both positive and negative infinity are present, the sum will be Not
    A Number (NaN).

    Examples
    --------
    >>> import numpy as np
    >>> np.nansum(1)
    1
    >>> np.nansum([1])
    1
    >>> np.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> np.nansum(a)
    3.0
    >>> np.nansum(a, axis=0)
    array([2.,  1.])
    >>> np.nansum([1, np.nan, np.inf])
    inf
    >>> np.nansum([1, np.nan, -np.inf])
    -inf
    >>> from numpy.testing import suppress_warnings
    >>> with np.errstate(invalid="ignore"):
    ...     np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present
    np.float64(nan)

    """
    a, mask = _replace_nan(a, 0)
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                  initial=initial, where=where)


def _nanprod_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                        initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nanprod_dispatcher)
def nanprod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
            initial=np._NoValue, where=np._NoValue):
    """
    Return the product of array elements over a given axis treating Not a
    Numbers (NaNs) as ones.

    One is returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose product is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the product is computed. The default is to compute
        the product of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will
        broadcast correctly against the original `arr`.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to include in the product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    numpy.prod : Product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> np.nanprod(1)
    1
    >>> np.nanprod([1])
    1
    >>> np.nanprod([1, np.nan])
    1.0
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanprod(a)
    6.0
    >>> np.nanprod(a, axis=0)
    array([3., 2.])

    """
    a, mask = _replace_nan(a, 1)
    return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                   initial=initial, where=where)


def _nancumsum_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_nancumsum_dispatcher)
def nancumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.  The cumulative sum does not change when NaNs are
    encountered and leading NaNs are replaced by zeros.

    Zeros are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See :ref:`ufuncs-output-type` for
        more details.

    Returns
    -------
    nancumsum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    numpy.cumsum : Cumulative sum across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> np.nancumsum(1)
    array([1])
    >>> np.nancumsum([1])
    array([1])
    >>> np.nancumsum([1, np.nan])
    array([1.,  1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumsum(a)
    array([1.,  3.,  6.,  6.])
    >>> np.nancumsum(a, axis=0)
    array([[1.,  2.],
           [4.,  2.]])
    >>> np.nancumsum(a, axis=1)
    array([[1.,  3.],
           [3.,  3.]])

    """
    a, mask = _replace_nan(a, 0)
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)


def _nancumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_nancumprod_dispatcher)
def nancumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of array elements over a given axis treating Not a
    Numbers (NaNs) as one.  The cumulative product does not change when NaNs are
    encountered and leading NaNs are replaced by ones.

    Ones are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed.  By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If *dtype* is not specified, it
        defaults to the dtype of `a`, unless `a` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Returns
    -------
    nancumprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    numpy.cumprod : Cumulative product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> np.nancumprod(1)
    array([1])
    >>> np.nancumprod([1])
    array([1])
    >>> np.nancumprod([1, np.nan])
    array([1.,  1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumprod(a)
    array([1.,  2.,  6.,  6.])
    >>> np.nancumprod(a, axis=0)
    array([[1.,  2.],
           [3.,  2.]])
    >>> np.nancumprod(a, axis=1)
    array([[1.,  2.],
           [3.,  3.]])

    """
    a, mask = _replace_nan(a, 1)
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)


def _nanmean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                        *, where=None):
    return (a, out)


@array_function_dispatch(_nanmean_dispatcher)
def nanmean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
            *, where=np._NoValue):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the input
        dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32`.  Specifying a
    higher-precision accumulator using the `dtype` keyword can alleviate
    this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanmean(a)
    2.6666666666666665
    >>> np.nanmean(a, axis=0)
    array([2.,  4.])
    >>> np.nanmean(a, axis=1)
    array([1.,  3.5]) # may vary

    """
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                       where=where)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims,
                 where=where)
    tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)
    avg = _divide_by_count(tot, cnt, out=out)

    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        # NaN is the only possible bad value, so no further
        # action is needed to handle bad results.
    return avg


def _nanmedian1d(arr1d, overwrite_input=False):
    """
    Private function for rank 1 arrays. Compute the median ignoring NaNs.
    See nanmedian for parameter usage
    """
    arr1d_parsed, _, overwrite_input = _remove_nan_1d(
        arr1d, overwrite_input=overwrite_input,
    )

    if arr1d_parsed.size == 0:
        # Ensure that a nan-esque scalar of the appropriate type (and unit)
        # is returned for `timedelta64` and `complexfloating`
        return arr1d[-1]

    return np.median(arr1d_parsed, overwrite_input=overwrite_input)


def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanmedian for parameter usage

    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        else:
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        # for small medians use sort + indexing which is still faster than
        # apply_along_axis
        # benchmarked with shuffled (50, 50, x) containing a few NaN
        if a.shape[axis] < 600:
            return _nanmedian_small(a, axis, out, overwrite_input)
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        if out is not None:
            out[...] = result
        return result


def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    """
    sort + indexing median, faster for small medians along multiple
    dimensions due to the high overhead of apply_along_axis

    see nanmedian for parameter usage
    """
    a = np.ma.masked_array(a, np.isnan(a))
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=5)

    fill_value = np.timedelta64("NaT") if m.dtype.kind == "m" else np.nan
    if out is not None:
        out[...] = m.filled(fill_value)
        return out
    return m.filled(fill_value)


def _nanmedian_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)


@array_function_dispatch(_nanmedian_dispatcher)
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue):
    """
    Compute the median along the specified axis, while ignoring NaNs.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
        A sequence of axes is supported since version 1.9.0.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean, median, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,
    ``V_sorted[(N-1)/2]``, when ``N`` is odd and the average of the two
    middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10.0, 7, 4], [3, 2, 1]])
    >>> a[0, 1] = np.nan
    >>> a
    array([[10., nan,  4.],
           [ 3.,  2.,  1.]])
    >>> np.median(a)
    np.float64(nan)
    >>> np.nanmedian(a)
    3.0
    >>> np.nanmedian(a, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.median(a, axis=1)
    array([nan,  2.])
    >>> b = a.copy()
    >>> np.nanmedian(b, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.nanmedian(b, axis=None, overwrite_input=True)
    3.0
    >>> assert not np.all(a==b)

    """
    a = np.asanyarray(a)
    # apply_along_axis in _nanmedian doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)

    return fnb._ureduce(a, func=_nanmedian, keepdims=keepdims,
                        axis=axis, out=out,
                        overwrite_input=overwrite_input)


def _nanpercentile_dispatcher(
        a, q, axis=None, out=None, overwrite_input=None,
        method=None, keepdims=None, *, weights=None, interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_nanpercentile_dispatcher)
def nanpercentile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        *,
        weights=None,
        interpolation=None,
):
    """
    Compute the qth percentile of the data along the specified axis,
    while ignoring nan values.

    Returns the qth percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be
        between 0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default
        is to compute the percentile(s) along a flattened version of the
        array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the percentile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    nanmean
    nanmedian : equivalent to ``nanpercentile(..., 50)``
    percentile, median, mean
    nanquantile : equivalent to nanpercentile, except q in range [0, 1].

    Notes
    -----
    The behavior of `numpy.nanpercentile` with percentage `q` is that of
    `numpy.quantile` with argument ``q/100`` (ignoring nan values).
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = np.nan
    >>> a
    array([[10.,  nan,   4.],
          [ 3.,   2.,   1.]])
    >>> np.percentile(a, 50)
    np.float64(nan)
    >>> np.nanpercentile(a, 50)
    3.0
    >>> np.nanpercentile(a, 50, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.nanpercentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.nanpercentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.nanpercentile(a, 50, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5,  2. ,  2.5])

    >>> b = a.copy()
    >>> np.nanpercentile(b, 50, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    if interpolation is not None:
        method = fnb._check_interpolation_as_method(
            method, interpolation, "nanpercentile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100, out=...)
    if not fnb._quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _nanquantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                            method=None, keepdims=None, *, weights=None,
                            interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_nanquantile_dispatcher)
def nanquantile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        *,
        weights=None,
        interpolation=None,
):
    """
    Compute the qth quantile of the data along the specified axis,
    while ignoring nan values.
    Returns the qth quantile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored
    q : array_like of float
        Probability or sequence of probabilities for the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the quantile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single probability and `axis=None`, then the result
        is a scalar. If multiple probability levels are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    quantile
    nanmean, nanmedian
    nanmedian : equivalent to ``nanquantile(..., 0.5)``
    nanpercentile : same as nanquantile, but with q in the range [0, 100].

    Notes
    -----
    The behavior of `numpy.nanquantile` is the same as that of
    `numpy.quantile` (ignoring nan values).
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = np.nan
    >>> a
    array([[10.,  nan,   4.],
          [ 3.,   2.,   1.]])
    >>> np.quantile(a, 0.5)
    np.float64(nan)
    >>> np.nanquantile(a, 0.5)
    3.0
    >>> np.nanquantile(a, 0.5, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.nanquantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.nanquantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.nanquantile(a, 0.5, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5,  2. ,  2.5])
    >>> b = a.copy()
    >>> np.nanquantile(b, 0.5, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """

    if interpolation is not None:
        method = fnb._check_interpolation_as_method(
            method, interpolation, "nanquantile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float).
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    if not fnb._quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _nanquantile_unchecked(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        weights=None,
):
    """Assumes that q is in [0, 1], and is an ndarray"""
    # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    return fnb._ureduce(a,
                        func=_nanquantile_ureduce_func,
                        q=q,
                        weights=weights,
                        keepdims=keepdims,
                        axis=axis,
                        out=out,
                        overwrite_input=overwrite_input,
                        method=method)


def _nanquantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int | None = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        wgt = None if weights is None else weights.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, method, weights=wgt)
    # Note that this code could try to fill in `out` right away
    elif weights is None:
        result = np.apply_along_axis(_nanquantile_1d, axis, a, q,
                                     overwrite_input, method, weights)
        # apply_along_axis fills in collapsed axis with results.
        # Move those axes to the beginning to match percentile's
        # convention.
        if q.ndim != 0:
            from_ax = [axis + i for i in range(q.ndim)]
            result = np.moveaxis(result, from_ax, list(range(q.ndim)))
    else:
        # We need to apply along axis over 2 arrays, a and weights.
        # move operation axes to end for simplicity:
        a = np.moveaxis(a, axis, -1)
        if weights is not None:
            weights = np.moveaxis(weights, axis, -1)
        if out is not None:
            result = out
        else:
            # weights are limited to `inverted_cdf` so the result dtype
            # is known to be identical to that of `a` here:
            result = np.empty_like(a, shape=q.shape + a.shape[:-1])

        for ii in np.ndindex(a.shape[:-1]):
            result[(...,) + ii] = _nanquantile_1d(
                    a[ii], q, weights=weights[ii],
                    overwrite_input=overwrite_input, method=method,
            )
        # This path dealt with `out` already...
        return result

    if out is not None:
        out[...] = result
    return result


def _nanquantile_1d(
    arr1d, q, overwrite_input=False, method="linear", weights=None,
):
    """
    Private function for rank 1 arrays. Compute quantile ignoring NaNs.
    See nanpercentile for parameter usage
    """
    # TODO: What to do when arr1d = [1, np.nan] and weights = [0, 1]?
    arr1d, weights, overwrite_input = _remove_nan_1d(arr1d,
        second_arr1d=weights, overwrite_input=overwrite_input)
    if arr1d.size == 0:
        # convert to scalar
        return np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]

    return fnb._quantile_unchecked(
        arr1d,
        q,
        overwrite_input=overwrite_input,
        method=method,
        weights=weights,
    )


def _nanvar_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                       keepdims=None, *, where=None, mean=None,
                       correction=None):
    return (a, out)


@array_function_dispatch(_nanvar_dispatcher)
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue,
           *, where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of
    a distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float64`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : {int, float}, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
    where : array_like of bool, optional
        Elements to include in the variance. See `~numpy.ufunc.reduce` for
        details.

        .. versionadded:: 1.22.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this var function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If `out` is None, return a new array containing the variance,
        otherwise return a reference to the output array. If ddof is >= the
        number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd, nanmean
    :ref:`ufuncs-output-type`

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite
    population.  ``ddof=0`` provides a maximum likelihood estimate of the
    variance for normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    For this function to work on sub-classes of ndarray, they must define
    `sum` with the kwarg `keepdims`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanvar(a)
    1.5555555555555554
    >>> np.nanvar(a, axis=0)
    array([1.,  0.])
    >>> np.nanvar(a, axis=1)
    array([0.,  0.25])  # may vary

    """
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof,
                      keepdims=keepdims, where=where, mean=mean,
                      correction=correction)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    # Compute mean
    if type(arr) is np.matrix:
        _keepdims = np._NoValue
    else:
        _keepdims = True

    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=_keepdims,
                     where=where)

    if mean is not np._NoValue:
        avg = mean
    else:
        # we need to special case matrix for reverse compatibility
        # in order for this to work, these sums need to be called with
        # keepdims=True, however matrix now raises an error in this case, but
        # the reason that it drops the keepdims kwarg is to force keepdims=True
        # so this used to work by serendipity.
        avg = np.sum(arr, axis=axis, dtype=dtype,
                     keepdims=_keepdims, where=where)
        avg = _divide_by_count(avg, cnt)

    # Compute squared deviation from mean.
    np.subtract(arr, avg, out=arr, casting='unsafe', where=where)
    arr = _copyto(arr, 0, mask)
    if issubclass(arr.dtype.type, np.complexfloating):
        sqr = np.multiply(arr, arr.conj(), out=arr, where=where).real
    else:
        sqr = np.multiply(arr, arr, out=arr, where=where)

    # Compute variance.
    var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)

    # Precaution against reduced object arrays
    try:
        var_ndim = var.ndim
    except AttributeError:
        var_ndim = np.ndim(var)
    if var_ndim < cnt.ndim:
        # Subclasses of ndarray may ignore keepdims, so check here.
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    var = _divide_by_count(var, dof)

    isbad = (dof <= 0)
    if np.any(isbad):
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning,
                      stacklevel=2)
        # NaN, inf, or negative numbers are all possible bad
        # values, so explicitly replace them with NaN.
        var = _copyto(var, np.nan, isbad)
    return var


def _nanstd_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                       keepdims=None, *, where=None, mean=None,
                       correction=None):
    return (a, out)


@array_function_dispatch(_nanstd_dispatcher)
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue,
           *, where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    """
    Compute the standard deviation along the specified axis, while
    ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a
    distribution, of the non-NaN array elements. The standard deviation is
    computed for the flattened array by default, otherwise over the
    specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of the non-NaN values.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it
        is the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the
        calculated values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non-NaN
        elements.  By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If this value is anything but the default it is passed through
        as-is to the relevant functions of the sub-classes.  If these
        functions do not have a `keepdims` kwarg, a RuntimeError will
        be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this std function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard
        deviation, otherwise return a reference to the output array. If
        ddof is >= the number of non-NaN elements in a slice or the slice
        contains only NaNs, then the result for that slice is NaN.

    See Also
    --------
    var, mean, std
    nanvar, nanmean
    :ref:`ufuncs-output-type`

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is
    specified, the divisor ``N - ddof`` is used instead. In standard
    statistical practice, ``ddof=1`` provides an unbiased estimator of the
    variance of the infinite population. ``ddof=0`` provides a maximum
    likelihood estimate of the variance for normally distributed variables.
    The standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Note that, for complex numbers, `std` takes the absolute value before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the *std* is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example
    below).  Specifying a higher-accuracy accumulator using the `dtype`
    keyword can alleviate this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanstd(a)
    1.247219128924647
    >>> np.nanstd(a, axis=0)
    array([1., 0.])
    >>> np.nanstd(a, axis=1)
    array([0.,  0.5]) # may vary

    """
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims, where=where, mean=mean,
                 correction=correction)
    if isinstance(var, np.ndarray):
        std = np.sqrt(var, out=var)
    elif hasattr(var, 'dtype'):
        std = var.dtype.type(np.sqrt(var))
    else:
        std = np.sqrt(var)
    return std

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\recfunctions.py ===
"""
Collection of utilities to manipulate structured arrays.

Most of these functions were initially implemented by John Hunter for
matplotlib.  They have been rewritten and extended for convenience.

"""
import itertools

import numpy as np
import numpy.ma as ma
import numpy.ma.mrecords as mrec
from numpy._core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like

__all__ = [
    'append_fields', 'apply_along_fields', 'assign_fields_by_name',
    'drop_fields', 'find_duplicates', 'flatten_descr',
    'get_fieldstructure', 'get_names', 'get_names_flat',
    'join_by', 'merge_arrays', 'rec_append_fields',
    'rec_drop_fields', 'rec_join', 'recursive_fill_fields',
    'rename_fields', 'repack_fields', 'require_fields',
    'stack_arrays', 'structured_to_unstructured', 'unstructured_to_structured',
    ]


def _recursive_fill_fields_dispatcher(input, output):
    return (input, output)


@array_function_dispatch(_recursive_fill_fields_dispatcher)
def recursive_fill_fields(input, output):
    """
    Fills fields from output with fields from input,
    with support for nested structures.

    Parameters
    ----------
    input : ndarray
        Input array.
    output : ndarray
        Output array.

    Notes
    -----
    * `output` should be at least the same size as `input`

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, 10.), (2, 20.)], dtype=[('A', np.int64), ('B', np.float64)])
    >>> b = np.zeros((3,), dtype=a.dtype)
    >>> rfn.recursive_fill_fields(a, b)
    array([(1, 10.), (2, 20.), (0,  0.)], dtype=[('A', '<i8'), ('B', '<f8')])

    """
    newdtype = output.dtype
    for field in newdtype.names:
        try:
            current = input[field]
        except ValueError:
            continue
        if current.dtype.names is not None:
            recursive_fill_fields(current, output[field])
        else:
            output[field][:len(current)] = current
    return output


def _get_fieldspec(dtype):
    """
    Produce a list of name/dtype pairs corresponding to the dtype fields

    Similar to dtype.descr, but the second item of each tuple is a dtype, not a
    string. As a result, this handles subarray dtypes

    Can be passed to the dtype constructor to reconstruct the dtype, noting that
    this (deliberately) discards field offsets.

    Examples
    --------
    >>> import numpy as np
    >>> dt = np.dtype([(('a', 'A'), np.int64), ('b', np.double, 3)])
    >>> dt.descr
    [(('a', 'A'), '<i8'), ('b', '<f8', (3,))]
    >>> _get_fieldspec(dt)
    [(('a', 'A'), dtype('int64')), ('b', dtype(('<f8', (3,))))]

    """
    if dtype.names is None:
        # .descr returns a nameless field, so we should too
        return [('', dtype)]
    else:
        fields = ((name, dtype.fields[name]) for name in dtype.names)
        # keep any titles, if present
        return [
            (name if len(f) == 2 else (f[2], name), f[0])
            for name, f in fields
        ]


def get_names(adtype):
    """
    Returns the field names of the input datatype as a tuple. Input datatype
    must have fields otherwise error is raised.

    Parameters
    ----------
    adtype : dtype
        Input datatype

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> rfn.get_names(np.empty((1,), dtype=[('A', int)]).dtype)
    ('A',)
    >>> rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]).dtype)
    ('A', 'B')
    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
    >>> rfn.get_names(adtype)
    ('a', ('b', ('ba', 'bb')))
    """
    listnames = []
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            listnames.append((name, tuple(get_names(current))))
        else:
            listnames.append(name)
    return tuple(listnames)


def get_names_flat(adtype):
    """
    Returns the field names of the input datatype as a tuple. Input datatype
    must have fields otherwise error is raised.
    Nested structure are flattened beforehand.

    Parameters
    ----------
    adtype : dtype
        Input datatype

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A', int)]).dtype) is None
    False
    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', str)]).dtype)
    ('A', 'B')
    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
    >>> rfn.get_names_flat(adtype)
    ('a', 'b', 'ba', 'bb')
    """
    listnames = []
    names = adtype.names
    for name in names:
        listnames.append(name)
        current = adtype[name]
        if current.names is not None:
            listnames.extend(get_names_flat(current))
    return tuple(listnames)


def flatten_descr(ndtype):
    """
    Flatten a structured data-type description.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])
    >>> rfn.flatten_descr(ndtype)
    (('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))

    """
    names = ndtype.names
    if names is None:
        return (('', ndtype),)
    else:
        descr = []
        for field in names:
            (typ, _) = ndtype.fields[field]
            if typ.names is not None:
                descr.extend(flatten_descr(typ))
            else:
                descr.append((field, typ))
        return tuple(descr)


def _zip_dtype(seqarrays, flatten=False):
    newdtype = []
    if flatten:
        for a in seqarrays:
            newdtype.extend(flatten_descr(a.dtype))
    else:
        for a in seqarrays:
            current = a.dtype
            if current.names is not None and len(current.names) == 1:
                # special case - dtypes of 1 field are flattened
                newdtype.extend(_get_fieldspec(current))
            else:
                newdtype.append(('', current))
    return np.dtype(newdtype)


def _zip_descr(seqarrays, flatten=False):
    """
    Combine the dtype description of a series of arrays.

    Parameters
    ----------
    seqarrays : sequence of arrays
        Sequence of arrays
    flatten : {boolean}, optional
        Whether to collapse nested descriptions.
    """
    return _zip_dtype(seqarrays, flatten=flatten).descr


def get_fieldstructure(adtype, lastname=None, parents=None,):
    """
    Returns a dictionary with fields indexing lists of their parent fields.

    This function is used to simplify access to fields nested in other fields.

    Parameters
    ----------
    adtype : np.dtype
        Input datatype
    lastname : optional
        Last processed field name (used internally during recursion).
    parents : dictionary
        Dictionary of parent fields (used internally during recursion).

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> ndtype =  np.dtype([('A', int),
    ...                     ('B', [('BA', int),
    ...                            ('BB', [('BBA', int), ('BBB', int)])])])
    >>> rfn.get_fieldstructure(ndtype)
    ... # XXX: possible regression, order of BBA and BBB is swapped
    {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}

    """
    if parents is None:
        parents = {}
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            if lastname:
                parents[name] = [lastname, ]
            else:
                parents[name] = []
            parents.update(get_fieldstructure(current, name, parents))
        else:
            lastparent = list(parents.get(lastname, []) or [])
            if lastparent:
                lastparent.append(lastname)
            elif lastname:
                lastparent = [lastname, ]
            parents[name] = lastparent or []
    return parents


def _izip_fields_flat(iterable):
    """
    Returns an iterator of concatenated fields from a sequence of arrays,
    collapsing any nested structure.

    """
    for element in iterable:
        if isinstance(element, np.void):
            yield from _izip_fields_flat(tuple(element))
        else:
            yield element


def _izip_fields(iterable):
    """
    Returns an iterator of concatenated fields from a sequence of arrays.

    """
    for element in iterable:
        if (hasattr(element, '__iter__') and
                not isinstance(element, str)):
            yield from _izip_fields(element)
        elif isinstance(element, np.void) and len(tuple(element)) == 1:
            # this statement is the same from the previous expression
            yield from _izip_fields(element)
        else:
            yield element


def _izip_records(seqarrays, fill_value=None, flatten=True):
    """
    Returns an iterator of concatenated items from a sequence of arrays.

    Parameters
    ----------
    seqarrays : sequence of arrays
        Sequence of arrays.
    fill_value : {None, integer}
        Value used to pad shorter iterables.
    flatten : {True, False},
        Whether to
    """

    # Should we flatten the items, or just use a nested approach
    if flatten:
        zipfunc = _izip_fields_flat
    else:
        zipfunc = _izip_fields

    for tup in itertools.zip_longest(*seqarrays, fillvalue=fill_value):
        yield tuple(zipfunc(tup))


def _fix_output(output, usemask=True, asrecarray=False):
    """
    Private function: return a recarray, a ndarray, a MaskedArray
    or a MaskedRecords depending on the input parameters
    """
    if not isinstance(output, ma.MaskedArray):
        usemask = False
    if usemask:
        if asrecarray:
            output = output.view(mrec.MaskedRecords)
    else:
        output = ma.filled(output)
        if asrecarray:
            output = output.view(np.recarray)
    return output


def _fix_defaults(output, defaults=None):
    """
    Update the fill_value and masked data of `output`
    from the default given in a dictionary defaults.
    """
    names = output.dtype.names
    (data, mask, fill_value) = (output.data, output.mask, output.fill_value)
    for (k, v) in (defaults or {}).items():
        if k in names:
            fill_value[k] = v
            data[k][mask[k]] = v
    return output


def _merge_arrays_dispatcher(seqarrays, fill_value=None, flatten=None,
                             usemask=None, asrecarray=None):
    return seqarrays


@array_function_dispatch(_merge_arrays_dispatcher)
def merge_arrays(seqarrays, fill_value=-1, flatten=False,
                 usemask=False, asrecarray=False):
    """
    Merge arrays field by field.

    Parameters
    ----------
    seqarrays : sequence of ndarrays
        Sequence of arrays
    fill_value : {float}, optional
        Filling value used to pad missing data on the shorter arrays.
    flatten : {False, True}, optional
        Whether to collapse nested fields.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : {False, True}, optional
        Whether to return a recarray (MaskedRecords) or not.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])))
    array([( 1, 10.), ( 2, 20.), (-1, 30.)],
          dtype=[('f0', '<i8'), ('f1', '<f8')])

    >>> rfn.merge_arrays((np.array([1, 2], dtype=np.int64),
    ...         np.array([10., 20., 30.])), usemask=False)
     array([(1, 10.0), (2, 20.0), (-1, 30.0)],
             dtype=[('f0', '<i8'), ('f1', '<f8')])
    >>> rfn.merge_arrays((np.array([1, 2]).view([('a', np.int64)]),
    ...               np.array([10., 20., 30.])),
    ...              usemask=False, asrecarray=True)
    rec.array([( 1, 10.), ( 2, 20.), (-1, 30.)],
              dtype=[('a', '<i8'), ('f1', '<f8')])

    Notes
    -----
    * Without a mask, the missing value will be filled with something,
      depending on what its corresponding type:

      * ``-1``      for integers
      * ``-1.0``    for floating point numbers
      * ``'-'``     for characters
      * ``'-1'``    for strings
      * ``True``    for boolean values
    * XXX: I just obtained these values empirically
    """
    # Only one item in the input sequence ?
    if (len(seqarrays) == 1):
        seqarrays = np.asanyarray(seqarrays[0])
    # Do we have a single ndarray as input ?
    if isinstance(seqarrays, (np.ndarray, np.void)):
        seqdtype = seqarrays.dtype
        # Make sure we have named fields
        if seqdtype.names is None:
            seqdtype = np.dtype([('', seqdtype)])
        if not flatten or _zip_dtype((seqarrays,), flatten=True) == seqdtype:
            # Minimal processing needed: just make sure everything's a-ok
            seqarrays = seqarrays.ravel()
            # Find what type of array we must return
            if usemask:
                if asrecarray:
                    seqtype = mrec.MaskedRecords
                else:
                    seqtype = ma.MaskedArray
            elif asrecarray:
                seqtype = np.recarray
            else:
                seqtype = np.ndarray
            return seqarrays.view(dtype=seqdtype, type=seqtype)
        else:
            seqarrays = (seqarrays,)
    else:
        # Make sure we have arrays in the input sequence
        seqarrays = [np.asanyarray(_m) for _m in seqarrays]
    # Find the sizes of the inputs and their maximum
    sizes = tuple(a.size for a in seqarrays)
    maxlength = max(sizes)
    # Get the dtype of the output (flattening if needed)
    newdtype = _zip_dtype(seqarrays, flatten=flatten)
    # Initialize the sequences for data and mask
    seqdata = []
    seqmask = []
    # If we expect some kind of MaskedArray, make a special loop.
    if usemask:
        for (a, n) in zip(seqarrays, sizes):
            nbmissing = (maxlength - n)
            # Get the data and mask
            data = a.ravel().__array__()
            mask = ma.getmaskarray(a).ravel()
            # Get the filling value (if needed)
            if nbmissing:
                fval = mrec._check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (np.ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                        fmsk = True
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
                        fmsk = np.ones((1,), dtype=mask.dtype)
            else:
                fval = None
                fmsk = True
            # Store an iterator padding the input to the expected length
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
            seqmask.append(itertools.chain(mask, [fmsk] * nbmissing))
        # Create an iterator for the data
        data = tuple(_izip_records(seqdata, flatten=flatten))
        output = ma.array(np.fromiter(data, dtype=newdtype, count=maxlength),
                          mask=list(_izip_records(seqmask, flatten=flatten)))
        if asrecarray:
            output = output.view(mrec.MaskedRecords)
    else:
        # Same as before, without the mask we don't need...
        for (a, n) in zip(seqarrays, sizes):
            nbmissing = (maxlength - n)
            data = a.ravel().__array__()
            if nbmissing:
                fval = mrec._check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (np.ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
            else:
                fval = None
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
        output = np.fromiter(tuple(_izip_records(seqdata, flatten=flatten)),
                             dtype=newdtype, count=maxlength)
        if asrecarray:
            output = output.view(np.recarray)
    # And we're done...
    return output


def _drop_fields_dispatcher(base, drop_names, usemask=None, asrecarray=None):
    return (base,)


@array_function_dispatch(_drop_fields_dispatcher)
def drop_fields(base, drop_names, usemask=True, asrecarray=False):
    """
    Return a new array with fields in `drop_names` dropped.

    Nested fields are supported.

    Parameters
    ----------
    base : array
        Input array
    drop_names : string or sequence
        String or sequence of strings corresponding to the names of the
        fields to drop.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : string or sequence, optional
        Whether to return a recarray or a mrecarray (`asrecarray=True`) or
        a plain ndarray or masked array with flexible dtype. The default
        is False.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
    ...   dtype=[('a', np.int64), ('b', [('ba', np.double), ('bb', np.int64)])])
    >>> rfn.drop_fields(a, 'a')
    array([((2., 3),), ((5., 6),)],
          dtype=[('b', [('ba', '<f8'), ('bb', '<i8')])])
    >>> rfn.drop_fields(a, 'ba')
    array([(1, (3,)), (4, (6,))], dtype=[('a', '<i8'), ('b', [('bb', '<i8')])])
    >>> rfn.drop_fields(a, ['ba', 'bb'])
    array([(1,), (4,)], dtype=[('a', '<i8')])
    """
    if _is_string_like(drop_names):
        drop_names = [drop_names]
    else:
        drop_names = set(drop_names)

    def _drop_descr(ndtype, drop_names):
        names = ndtype.names
        newdtype = []
        for name in names:
            current = ndtype[name]
            if name in drop_names:
                continue
            if current.names is not None:
                descr = _drop_descr(current, drop_names)
                if descr:
                    newdtype.append((name, descr))
            else:
                newdtype.append((name, current))
        return newdtype

    newdtype = _drop_descr(base.dtype, drop_names)

    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


def _keep_fields(base, keep_names, usemask=True, asrecarray=False):
    """
    Return a new array keeping only the fields in `keep_names`,
    and preserving the order of those fields.

    Parameters
    ----------
    base : array
        Input array
    keep_names : string or sequence
        String or sequence of strings corresponding to the names of the
        fields to keep. Order of the names will be preserved.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : string or sequence, optional
        Whether to return a recarray or a mrecarray (`asrecarray=True`) or
        a plain ndarray or masked array with flexible dtype. The default
        is False.
    """
    newdtype = [(n, base.dtype[n]) for n in keep_names]
    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


def _rec_drop_fields_dispatcher(base, drop_names):
    return (base,)


@array_function_dispatch(_rec_drop_fields_dispatcher)
def rec_drop_fields(base, drop_names):
    """
    Returns a new numpy.recarray with fields in `drop_names` dropped.
    """
    return drop_fields(base, drop_names, usemask=False, asrecarray=True)


def _rename_fields_dispatcher(base, namemapper):
    return (base,)


@array_function_dispatch(_rename_fields_dispatcher)
def rename_fields(base, namemapper):
    """
    Rename the fields from a flexible-datatype ndarray or recarray.

    Nested fields are supported.

    Parameters
    ----------
    base : ndarray
        Input array whose fields must be modified.
    namemapper : dictionary
        Dictionary mapping old field names to their new version.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
    ...   dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])
    >>> rfn.rename_fields(a, {'a':'A', 'bb':'BB'})
    array([(1, (2., [ 3., 30.])), (4, (5., [ 6., 60.]))],
          dtype=[('A', '<i8'), ('b', [('ba', '<f8'), ('BB', '<f8', (2,))])])

    """
    def _recursive_rename_fields(ndtype, namemapper):
        newdtype = []
        for name in ndtype.names:
            newname = namemapper.get(name, name)
            current = ndtype[name]
            if current.names is not None:
                newdtype.append(
                    (newname, _recursive_rename_fields(current, namemapper))
                    )
            else:
                newdtype.append((newname, current))
        return newdtype
    newdtype = _recursive_rename_fields(base.dtype, namemapper)
    return base.view(newdtype)


def _append_fields_dispatcher(base, names, data, dtypes=None,
                              fill_value=None, usemask=None, asrecarray=None):
    yield base
    yield from data


@array_function_dispatch(_append_fields_dispatcher)
def append_fields(base, names, data, dtypes=None,
                  fill_value=-1, usemask=True, asrecarray=False):
    """
    Add new fields to an existing array.

    The names of the fields are given with the `names` arguments,
    the corresponding values with the `data` arguments.
    If a single field is appended, `names`, `data` and `dtypes` do not have
    to be lists but just values.

    Parameters
    ----------
    base : array
        Input array to extend.
    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new fields.
    data : array or sequence of arrays
        Array or sequence of arrays storing the fields to add to the base.
    dtypes : sequence of datatypes, optional
        Datatype or sequence of datatypes.
        If None, the datatypes are estimated from the `data`.
    fill_value : {float}, optional
        Filling value used to pad missing data on the shorter arrays.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : {False, True}, optional
        Whether to return a recarray (MaskedRecords) or not.

    """
    # Check the names
    if isinstance(names, (tuple, list)):
        if len(names) != len(data):
            msg = "The number of arrays does not match the number of names"
            raise ValueError(msg)
    elif isinstance(names, str):
        names = [names, ]
        data = [data, ]
    #
    if dtypes is None:
        data = [np.array(a, copy=None, subok=True) for a in data]
        data = [a.view([(name, a.dtype)]) for (name, a) in zip(names, data)]
    else:
        if not isinstance(dtypes, (tuple, list)):
            dtypes = [dtypes, ]
        if len(data) != len(dtypes):
            if len(dtypes) == 1:
                dtypes = dtypes * len(data)
            else:
                msg = "The dtypes argument must be None, a dtype, or a list."
                raise ValueError(msg)
        data = [np.array(a, copy=None, subok=True, dtype=d).view([(n, d)])
                for (a, n, d) in zip(data, names, dtypes)]
    #
    base = merge_arrays(base, usemask=usemask, fill_value=fill_value)
    if len(data) > 1:
        data = merge_arrays(data, flatten=True, usemask=usemask,
                            fill_value=fill_value)
    else:
        data = data.pop()
    #
    output = ma.masked_all(
        max(len(base), len(data)),
        dtype=_get_fieldspec(base.dtype) + _get_fieldspec(data.dtype))
    output = recursive_fill_fields(base, output)
    output = recursive_fill_fields(data, output)
    #
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


def _rec_append_fields_dispatcher(base, names, data, dtypes=None):
    yield base
    yield from data


@array_function_dispatch(_rec_append_fields_dispatcher)
def rec_append_fields(base, names, data, dtypes=None):
    """
    Add new fields to an existing array.

    The names of the fields are given with the `names` arguments,
    the corresponding values with the `data` arguments.
    If a single field is appended, `names`, `data` and `dtypes` do not have
    to be lists but just values.

    Parameters
    ----------
    base : array
        Input array to extend.
    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new fields.
    data : array or sequence of arrays
        Array or sequence of arrays storing the fields to add to the base.
    dtypes : sequence of datatypes, optional
        Datatype or sequence of datatypes.
        If None, the datatypes are estimated from the `data`.

    See Also
    --------
    append_fields

    Returns
    -------
    appended_array : np.recarray
    """
    return append_fields(base, names, data=data, dtypes=dtypes,
                         asrecarray=True, usemask=False)


def _repack_fields_dispatcher(a, align=None, recurse=None):
    return (a,)


@array_function_dispatch(_repack_fields_dispatcher)
def repack_fields(a, align=False, recurse=False):
    """
    Re-pack the fields of a structured array or dtype in memory.

    The memory layout of structured datatypes allows fields at arbitrary
    byte offsets. This means the fields can be separated by padding bytes,
    their offsets can be non-monotonically increasing, and they can overlap.

    This method removes any overlaps and reorders the fields in memory so they
    have increasing byte offsets, and adds or removes padding bytes depending
    on the `align` option, which behaves like the `align` option to
    `numpy.dtype`.

    If `align=False`, this method produces a "packed" memory layout in which
    each field starts at the byte the previous field ended, and any padding
    bytes are removed.

    If `align=True`, this methods produces an "aligned" memory layout in which
    each field's offset is a multiple of its alignment, and the total itemsize
    is a multiple of the largest alignment, by adding padding bytes as needed.

    Parameters
    ----------
    a : ndarray or dtype
       array or dtype for which to repack the fields.
    align : boolean
       If true, use an "aligned" memory layout, otherwise use a "packed" layout.
    recurse : boolean
       If True, also repack nested structures.

    Returns
    -------
    repacked : ndarray or dtype
       Copy of `a` with fields repacked, or `a` itself if no repacking was
       needed.

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.lib import recfunctions as rfn
    >>> def print_offsets(d):
    ...     print("offsets:", [d.fields[name][1] for name in d.names])
    ...     print("itemsize:", d.itemsize)
    ...
    >>> dt = np.dtype('u1, <i8, <f8', align=True)
    >>> dt
    dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['u1', '<i8', '<f8'], \
'offsets': [0, 8, 16], 'itemsize': 24}, align=True)
    >>> print_offsets(dt)
    offsets: [0, 8, 16]
    itemsize: 24
    >>> packed_dt = rfn.repack_fields(dt)
    >>> packed_dt
    dtype([('f0', 'u1'), ('f1', '<i8'), ('f2', '<f8')])
    >>> print_offsets(packed_dt)
    offsets: [0, 1, 9]
    itemsize: 17

    """
    if not isinstance(a, np.dtype):
        dt = repack_fields(a.dtype, align=align, recurse=recurse)
        return a.astype(dt, copy=False)

    if a.names is None:
        return a

    fieldinfo = []
    for name in a.names:
        tup = a.fields[name]
        if recurse:
            fmt = repack_fields(tup[0], align=align, recurse=True)
        else:
            fmt = tup[0]

        if len(tup) == 3:
            name = (tup[2], name)

        fieldinfo.append((name, fmt))

    dt = np.dtype(fieldinfo, align=align)
    return np.dtype((a.type, dt))

def _get_fields_and_offsets(dt, offset=0):
    """
    Returns a flat list of (dtype, count, offset) tuples of all the
    scalar fields in the dtype "dt", including nested fields, in left
    to right order.
    """

    # counts up elements in subarrays, including nested subarrays, and returns
    # base dtype and count
    def count_elem(dt):
        count = 1
        while dt.shape != ():
            for size in dt.shape:
                count *= size
            dt = dt.base
        return dt, count

    fields = []
    for name in dt.names:
        field = dt.fields[name]
        f_dt, f_offset = field[0], field[1]
        f_dt, n = count_elem(f_dt)

        if f_dt.names is None:
            fields.append((np.dtype((f_dt, (n,))), n, f_offset + offset))
        else:
            subfields = _get_fields_and_offsets(f_dt, f_offset + offset)
            size = f_dt.itemsize

            for i in range(n):
                if i == 0:
                    # optimization: avoid list comprehension if no subarray
                    fields.extend(subfields)
                else:
                    fields.extend([(d, c, o + i * size) for d, c, o in subfields])
    return fields

def _common_stride(offsets, counts, itemsize):
    """
    Returns the stride between the fields, or None if the stride is not
    constant. The values in "counts" designate the lengths of
    subarrays. Subarrays are treated as many contiguous fields, with
    always positive stride.
    """
    if len(offsets) <= 1:
        return itemsize

    negative = offsets[1] < offsets[0]  # negative stride
    if negative:
        # reverse, so offsets will be ascending
        it = zip(reversed(offsets), reversed(counts))
    else:
        it = zip(offsets, counts)

    prev_offset = None
    stride = None
    for offset, count in it:
        if count != 1:  # subarray: always c-contiguous
            if negative:
                return None  # subarrays can never have a negative stride
            if stride is None:
                stride = itemsize
            if stride != itemsize:
                return None
            end_offset = offset + (count - 1) * itemsize
        else:
            end_offset = offset

        if prev_offset is not None:
            new_stride = offset - prev_offset
            if stride is None:
                stride = new_stride
            if stride != new_stride:
                return None

        prev_offset = end_offset

    if negative:
        return -stride
    return stride


def _structured_to_unstructured_dispatcher(arr, dtype=None, copy=None,
                                           casting=None):
    return (arr,)

@array_function_dispatch(_structured_to_unstructured_dispatcher)
def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    """
    Converts an n-D structured array into an (n+1)-D unstructured array.

    The new array will have a new last dimension equal in size to the
    number of field-elements of the input array. If not supplied, the output
    datatype is determined from the numpy type promotion rules applied to all
    the field datatypes.

    Nested fields, as well as each element of any subarray fields, all count
    as a single field-elements.

    Parameters
    ----------
    arr : ndarray
       Structured array or dtype to convert. Cannot contain object datatype.
    dtype : dtype, optional
       The dtype of the output unstructured array.
    copy : bool, optional
        If true, always return a copy. If false, a view is returned if
        possible, such as when the `dtype` and strides of the fields are
        suitable and the array subtype is one of `numpy.ndarray`,
        `numpy.recarray` or `numpy.memmap`.

        .. versionchanged:: 1.25.0
            A view can now be returned if the fields are separated by a
            uniform stride.

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `numpy.ndarray.astype`. Controls what kind of
        data casting may occur.

    Returns
    -------
    unstructured : ndarray
       Unstructured array with one more dimension.

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
    >>> a
    array([(0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.]),
           (0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.])],
          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])
    >>> rfn.structured_to_unstructured(a)
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
    ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
    >>> np.mean(rfn.structured_to_unstructured(b[['x', 'z']]), axis=-1)
    array([ 3. ,  5.5,  9. , 11. ])

    """  # noqa: E501
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    fields = _get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    if n_fields == 0 and dtype is None:
        raise ValueError("arr has no fields. Unable to guess dtype")
    elif n_fields == 0:
        # too many bugs elsewhere for this to work now
        raise NotImplementedError("arr with no fields is not supported")

    dts, counts, offsets = zip(*fields)
    names = [f'f{n}' for n in range(n_fields)]

    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = np.dtype(dtype)

    # Use a series of views and casts to convert to an unstructured array:

    # first view using flattened fields (doesn't work for object arrays)
    # Note: dts may include a shape for subarrays
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': arr.dtype.itemsize})
    arr = arr.view(flattened_fields)

    # we only allow a few types to be unstructured by manipulating the
    # strides, because we know it won't work with, for example, np.matrix nor
    # np.ma.MaskedArray.
    can_view = type(arr) in (np.ndarray, np.recarray, np.memmap)
    if (not copy) and can_view and all(dt.base == out_dtype for dt in dts):
        # all elements have the right dtype already; if they have a common
        # stride, we can just return a view
        common_stride = _common_stride(offsets, counts, out_dtype.itemsize)
        if common_stride is not None:
            wrap = arr.__array_wrap__

            new_shape = arr.shape + (sum(counts), out_dtype.itemsize)
            new_strides = arr.strides + (abs(common_stride), 1)

            arr = arr[..., np.newaxis].view(np.uint8)  # view as bytes
            arr = arr[..., min(offsets):]  # remove the leading unused data
            arr = np.lib.stride_tricks.as_strided(arr,
                                                  new_shape,
                                                  new_strides,
                                                  subok=True)

            # cast and drop the last dimension again
            arr = arr.view(out_dtype)[..., 0]

            if common_stride < 0:
                arr = arr[..., ::-1]  # reverse, if the stride was negative
            if type(arr) is not type(wrap.__self__):
                # Some types (e.g. recarray) turn into an ndarray along the
                # way, so we have to wrap it again in order to match the
                # behavior with copy=True.
                arr = wrap(arr)
            return arr

    # next cast to a packed format with all fields converted to new dtype
    packed_fields = np.dtype({'names': names,
                              'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)

    # finally is it safe to view the packed fields as the unstructured type
    return arr.view((out_dtype, (sum(counts),)))


def _unstructured_to_structured_dispatcher(arr, dtype=None, names=None,
                                           align=None, copy=None, casting=None):
    return (arr,)

@array_function_dispatch(_unstructured_to_structured_dispatcher)
def unstructured_to_structured(arr, dtype=None, names=None, align=False,
                               copy=False, casting='unsafe'):
    """
    Converts an n-D unstructured array into an (n-1)-D structured array.

    The last dimension of the input array is converted into a structure, with
    number of field-elements equal to the size of the last dimension of the
    input array. By default all output fields have the input array's dtype, but
    an output structured dtype with an equal number of fields-elements can be
    supplied instead.

    Nested fields, as well as each element of any subarray fields, all count
    towards the number of field-elements.

    Parameters
    ----------
    arr : ndarray
       Unstructured array or dtype to convert.
    dtype : dtype, optional
       The structured dtype of the output array
    names : list of strings, optional
       If dtype is not supplied, this specifies the field names for the output
       dtype, in order. The field dtypes will be the same as the input array.
    align : boolean, optional
       Whether to create an aligned memory layout.
    copy : bool, optional
        See copy argument to `numpy.ndarray.astype`. If true, always return a
        copy. If false, and `dtype` requirements are satisfied, a view is
        returned.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `numpy.ndarray.astype`. Controls what kind of
        data casting may occur.

    Returns
    -------
    structured : ndarray
       Structured array with fewer dimensions.

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.lib import recfunctions as rfn
    >>> dt = np.dtype([('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
    >>> a = np.arange(20).reshape((4,5))
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    >>> rfn.unstructured_to_structured(a, dt)
    array([( 0, ( 1.,  2), [ 3.,  4.]), ( 5, ( 6.,  7), [ 8.,  9.]),
           (10, (11., 12), [13., 14.]), (15, (16., 17), [18., 19.])],
          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])

    """  # noqa: E501
    if arr.shape == ():
        raise ValueError('arr must have at least one dimension')
    n_elem = arr.shape[-1]
    if n_elem == 0:
        # too many bugs elsewhere for this to work now
        raise NotImplementedError("last axis with size 0 is not supported")

    if dtype is None:
        if names is None:
            names = [f'f{n}' for n in range(n_elem)]
        out_dtype = np.dtype([(n, arr.dtype) for n in names], align=align)
        fields = _get_fields_and_offsets(out_dtype)
        dts, counts, offsets = zip(*fields)
    else:
        if names is not None:
            raise ValueError("don't supply both dtype and names")
        # if dtype is the args of np.dtype, construct it
        dtype = np.dtype(dtype)
        # sanity check of the input dtype
        fields = _get_fields_and_offsets(dtype)
        if len(fields) == 0:
            dts, counts, offsets = [], [], []
        else:
            dts, counts, offsets = zip(*fields)

        if n_elem != sum(counts):
            raise ValueError('The length of the last dimension of arr must '
                             'be equal to the number of fields in dtype')
        out_dtype = dtype
        if align and not out_dtype.isalignedstruct:
            raise ValueError("align was True but dtype is not aligned")

    names = [f'f{n}' for n in range(len(fields))]

    # Use a series of views and casts to convert to a structured array:

    # first view as a packed structured array of one dtype
    packed_fields = np.dtype({'names': names,
                              'formats': [(arr.dtype, dt.shape) for dt in dts]})
    arr = np.ascontiguousarray(arr).view(packed_fields)

    # next cast to an unpacked but flattened format with varied dtypes
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': out_dtype.itemsize})
    arr = arr.astype(flattened_fields, copy=copy, casting=casting)

    # finally view as the final nested dtype and remove the last axis
    return arr.view(out_dtype)[..., 0]

def _apply_along_fields_dispatcher(func, arr):
    return (arr,)

@array_function_dispatch(_apply_along_fields_dispatcher)
def apply_along_fields(func, arr):
    """
    Apply function 'func' as a reduction across fields of a structured array.

    This is similar to `numpy.apply_along_axis`, but treats the fields of a
    structured array as an extra axis. The fields are all first cast to a
    common type following the type-promotion rules from `numpy.result_type`
    applied to the field's dtypes.

    Parameters
    ----------
    func : function
       Function to apply on the "field" dimension. This function must
       support an `axis` argument, like `numpy.mean`, `numpy.sum`, etc.
    arr : ndarray
       Structured array for which to apply func.

    Returns
    -------
    out : ndarray
       Result of the reduction operation

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.lib import recfunctions as rfn
    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
    ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
    >>> rfn.apply_along_fields(np.mean, b)
    array([ 2.66666667,  5.33333333,  8.66666667, 11.        ])
    >>> rfn.apply_along_fields(np.mean, b[['x', 'z']])
    array([ 3. ,  5.5,  9. , 11. ])

    """
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    uarr = structured_to_unstructured(arr)
    return func(uarr, axis=-1)
    # works and avoids axis requirement, but very, very slow:
    #return np.apply_along_axis(func, -1, uarr)

def _assign_fields_by_name_dispatcher(dst, src, zero_unassigned=None):
    return dst, src

@array_function_dispatch(_assign_fields_by_name_dispatcher)
def assign_fields_by_name(dst, src, zero_unassigned=True):
    """
    Assigns values from one structured array to another by field name.

    Normally in numpy >= 1.14, assignment of one structured array to another
    copies fields "by position", meaning that the first field from the src is
    copied to the first field of the dst, and so on, regardless of field name.

    This function instead copies "by field name", such that fields in the dst
    are assigned from the identically named field in the src. This applies
    recursively for nested structures. This is how structure assignment worked
    in numpy >= 1.6 to <= 1.13.

    Parameters
    ----------
    dst : ndarray
    src : ndarray
        The source and destination arrays during assignment.
    zero_unassigned : bool, optional
        If True, fields in the dst for which there was no matching
        field in the src are filled with the value 0 (zero). This
        was the behavior of numpy <= 1.13. If False, those fields
        are not modified.
    """

    if dst.dtype.names is None:
        dst[...] = src
        return

    for name in dst.dtype.names:
        if name not in src.dtype.names:
            if zero_unassigned:
                dst[name] = 0
        else:
            assign_fields_by_name(dst[name], src[name],
                                  zero_unassigned)

def _require_fields_dispatcher(array, required_dtype):
    return (array,)

@array_function_dispatch(_require_fields_dispatcher)
def require_fields(array, required_dtype):
    """
    Casts a structured array to a new dtype using assignment by field-name.

    This function assigns from the old to the new array by name, so the
    value of a field in the output array is the value of the field with the
    same name in the source array. This has the effect of creating a new
    ndarray containing only the fields "required" by the required_dtype.

    If a field name in the required_dtype does not exist in the
    input array, that field is created and set to 0 in the output array.

    Parameters
    ----------
    a : ndarray
       array to cast
    required_dtype : dtype
       datatype for output array

    Returns
    -------
    out : ndarray
        array with the new dtype, with field values copied from the fields in
        the input array with the same name

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.ones(4, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
    >>> rfn.require_fields(a, [('b', 'f4'), ('c', 'u1')])
    array([(1., 1), (1., 1), (1., 1), (1., 1)],
      dtype=[('b', '<f4'), ('c', 'u1')])
    >>> rfn.require_fields(a, [('b', 'f4'), ('newf', 'u1')])
    array([(1., 0), (1., 0), (1., 0), (1., 0)],
      dtype=[('b', '<f4'), ('newf', 'u1')])

    """
    out = np.empty(array.shape, dtype=required_dtype)
    assign_fields_by_name(out, array)
    return out


def _stack_arrays_dispatcher(arrays, defaults=None, usemask=None,
                             asrecarray=None, autoconvert=None):
    return arrays


@array_function_dispatch(_stack_arrays_dispatcher)
def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False,
                 autoconvert=False):
    """
    Superposes arrays fields by fields

    Parameters
    ----------
    arrays : array or sequence
        Sequence of input arrays.
    defaults : dictionary, optional
        Dictionary mapping field names to the corresponding default values.
    usemask : {True, False}, optional
        Whether to return a MaskedArray (or MaskedRecords is
        `asrecarray==True`) or a ndarray.
    asrecarray : {False, True}, optional
        Whether to return a recarray (or MaskedRecords if `usemask==True`)
        or just a flexible-type ndarray.
    autoconvert : {False, True}, optional
        Whether automatically cast the type of the field to the maximum.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> x = np.array([1, 2,])
    >>> rfn.stack_arrays(x) is x
    True
    >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
    >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
    ...   dtype=[('A', '|S3'), ('B', np.double), ('C', np.double)])
    >>> test = rfn.stack_arrays((z,zz))
    >>> test
    masked_array(data=[(b'A', 1.0, --), (b'B', 2.0, --), (b'a', 10.0, 100.0),
                       (b'b', 20.0, 200.0), (b'c', 30.0, 300.0)],
                 mask=[(False, False,  True), (False, False,  True),
                       (False, False, False), (False, False, False),
                       (False, False, False)],
           fill_value=(b'N/A', 1e+20, 1e+20),
                dtype=[('A', 'S3'), ('B', '<f8'), ('C', '<f8')])

    """
    if isinstance(arrays, np.ndarray):
        return arrays
    elif len(arrays) == 1:
        return arrays[0]
    seqarrays = [np.asanyarray(a).ravel() for a in arrays]
    nrecords = [len(a) for a in seqarrays]
    ndtype = [a.dtype for a in seqarrays]
    fldnames = [d.names for d in ndtype]
    #
    dtype_l = ndtype[0]
    newdescr = _get_fieldspec(dtype_l)
    names = [n for n, d in newdescr]
    for dtype_n in ndtype[1:]:
        for fname, fdtype in _get_fieldspec(dtype_n):
            if fname not in names:
                newdescr.append((fname, fdtype))
                names.append(fname)
            else:
                nameidx = names.index(fname)
                _, cdtype = newdescr[nameidx]
                if autoconvert:
                    newdescr[nameidx] = (fname, max(fdtype, cdtype))
                elif fdtype != cdtype:
                    raise TypeError(f"Incompatible type '{cdtype}' <> '{fdtype}'")
    # Only one field: use concatenate
    if len(newdescr) == 1:
        output = ma.concatenate(seqarrays)
    else:
        #
        output = ma.masked_all((np.sum(nrecords),), newdescr)
        offset = np.cumsum(np.r_[0, nrecords])
        seen = []
        for (a, n, i, j) in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
            names = a.dtype.names
            if names is None:
                output[f'f{len(seen)}'][i:j] = a
            else:
                for name in n:
                    output[name][i:j] = a[name]
                    if name not in seen:
                        seen.append(name)
    #
    return _fix_output(_fix_defaults(output, defaults),
                       usemask=usemask, asrecarray=asrecarray)


def _find_duplicates_dispatcher(
        a, key=None, ignoremask=None, return_index=None):
    return (a,)


@array_function_dispatch(_find_duplicates_dispatcher)
def find_duplicates(a, key=None, ignoremask=True, return_index=False):
    """
    Find the duplicates in a structured array along a given key

    Parameters
    ----------
    a : array-like
        Input array
    key : {string, None}, optional
        Name of the fields along which to check the duplicates.
        If None, the search is performed by records
    ignoremask : {True, False}, optional
        Whether masked data should be discarded or considered as duplicates.
    return_index : {False, True}, optional
        Whether to return the indices of the duplicated values.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import recfunctions as rfn
    >>> ndtype = [('a', int)]
    >>> a = np.ma.array([1, 1, 1, 2, 2, 3, 3],
    ...         mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
    >>> rfn.find_duplicates(a, ignoremask=True, return_index=True)
    (masked_array(data=[(1,), (1,), (2,), (2,)],
                 mask=[(False,), (False,), (False,), (False,)],
           fill_value=(999999,),
                dtype=[('a', '<i8')]), array([0, 1, 3, 4]))
    """
    a = np.asanyarray(a).ravel()
    # Get a dictionary of fields
    fields = get_fieldstructure(a.dtype)
    # Get the sorting data (by selecting the corresponding field)
    base = a
    if key:
        for f in fields[key]:
            base = base[f]
        base = base[key]
    # Get the sorting indices and the sorted data
    sortidx = base.argsort()
    sortedbase = base[sortidx]
    sorteddata = sortedbase.filled()
    # Compare the sorting data
    flag = (sorteddata[:-1] == sorteddata[1:])
    # If masked data must be ignored, set the flag to false where needed
    if ignoremask:
        sortedmask = sortedbase.recordmask
        flag[sortedmask[1:]] = False
    flag = np.concatenate(([False], flag))
    # We need to take the point on the left as well (else we're missing it)
    flag[:-1] = flag[:-1] + flag[1:]
    duplicates = a[sortidx][flag]
    if return_index:
        return (duplicates, sortidx[flag])
    else:
        return duplicates


def _join_by_dispatcher(
        key, r1, r2, jointype=None, r1postfix=None, r2postfix=None,
        defaults=None, usemask=None, asrecarray=None):
    return (r1, r2)


@array_function_dispatch(_join_by_dispatcher)
def join_by(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
            defaults=None, usemask=True, asrecarray=False):
    """
    Join arrays `r1` and `r2` on key `key`.

    The key should be either a string or a sequence of string corresponding
    to the fields used to join the array.  An exception is raised if the
    `key` field cannot be found in the two input arrays.  Neither `r1` nor
    `r2` should have any duplicates along `key`: the presence of duplicates
    will make the output quite unreliable. Note that duplicates are not
    looked for by the algorithm.

    Parameters
    ----------
    key : {string, sequence}
        A string or a sequence of strings corresponding to the fields used
        for comparison.
    r1, r2 : arrays
        Structured arrays.
    jointype : {'inner', 'outer', 'leftouter'}, optional
        If 'inner', returns the elements common to both r1 and r2.
        If 'outer', returns the common elements as well as the elements of
        r1 not in r2 and the elements of not in r2.
        If 'leftouter', returns the common elements and the elements of r1
        not in r2.
    r1postfix : string, optional
        String appended to the names of the fields of r1 that are present
        in r2 but absent of the key.
    r2postfix : string, optional
        String appended to the names of the fields of r2 that are present
        in r1 but absent of the key.
    defaults : {dictionary}, optional
        Dictionary mapping field names to the corresponding default values.
    usemask : {True, False}, optional
        Whether to return a MaskedArray (or MaskedRecords is
        `asrecarray==True`) or a ndarray.
    asrecarray : {False, True}, optional
        Whether to return a recarray (or MaskedRecords if `usemask==True`)
        or just a flexible-type ndarray.

    Notes
    -----
    * The output is sorted along the key.
    * A temporary array is formed by dropping the fields not in the key for
      the two arrays and concatenating the result. This array is then
      sorted, and the common entries selected. The output is constructed by
      filling the fields with the selected entries. Matching is not
      preserved if there are some duplicates...

    """
    # Check jointype
    if jointype not in ('inner', 'outer', 'leftouter'):
        raise ValueError(
                "The 'jointype' argument should be in 'inner', "
                "'outer' or 'leftouter' (got '%s' instead)" % jointype
                )
    # If we have a single key, put it in a tuple
    if isinstance(key, str):
        key = (key,)

    # Check the keys
    if len(set(key)) != len(key):
        dup = next(x for n, x in enumerate(key) if x in key[n + 1:])
        raise ValueError(f"duplicate join key {dup!r}")
    for name in key:
        if name not in r1.dtype.names:
            raise ValueError(f'r1 does not have key field {name!r}')
        if name not in r2.dtype.names:
            raise ValueError(f'r2 does not have key field {name!r}')

    # Make sure we work with ravelled arrays
    r1 = r1.ravel()
    r2 = r2.ravel()
    (nb1, nb2) = (len(r1), len(r2))
    (r1names, r2names) = (r1.dtype.names, r2.dtype.names)

    # Check the names for collision
    collisions = (set(r1names) & set(r2names)) - set(key)
    if collisions and not (r1postfix or r2postfix):
        msg = "r1 and r2 contain common names, r1postfix and r2postfix "
        msg += "can't both be empty"
        raise ValueError(msg)

    # Make temporary arrays of just the keys
    #  (use order of keys in `r1` for back-compatibility)
    key1 = [n for n in r1names if n in key]
    r1k = _keep_fields(r1, key1)
    r2k = _keep_fields(r2, key1)

    # Concatenate the two arrays for comparison
    aux = ma.concatenate((r1k, r2k))
    idx_sort = aux.argsort(order=key)
    aux = aux[idx_sort]
    #
    # Get the common keys
    flag_in = ma.concatenate(([False], aux[1:] == aux[:-1]))
    flag_in[:-1] = flag_in[1:] + flag_in[:-1]
    idx_in = idx_sort[flag_in]
    idx_1 = idx_in[(idx_in < nb1)]
    idx_2 = idx_in[(idx_in >= nb1)] - nb1
    (r1cmn, r2cmn) = (len(idx_1), len(idx_2))
    if jointype == 'inner':
        (r1spc, r2spc) = (0, 0)
    elif jointype == 'outer':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
        idx_2 = np.concatenate((idx_2, idx_out[(idx_out >= nb1)] - nb1))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, len(idx_2) - r2cmn)
    elif jointype == 'leftouter':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, 0)
    # Select the entries from each input
    (s1, s2) = (r1[idx_1], r2[idx_2])
    #
    # Build the new description of the output array .......
    # Start with the key fields
    ndtype = _get_fieldspec(r1k.dtype)

    # Add the fields from r1
    for fname, fdtype in _get_fieldspec(r1.dtype):
        if fname not in key:
            ndtype.append((fname, fdtype))

    # Add the fields from r2
    for fname, fdtype in _get_fieldspec(r2.dtype):
        # Have we seen the current name already ?
        # we need to rebuild this list every time
        names = [name for name, dtype in ndtype]
        try:
            nameidx = names.index(fname)
        except ValueError:
            #... we haven't: just add the description to the current list
            ndtype.append((fname, fdtype))
        else:
            # collision
            _, cdtype = ndtype[nameidx]
            if fname in key:
                # The current field is part of the key: take the largest dtype
                ndtype[nameidx] = (fname, max(fdtype, cdtype))
            else:
                # The current field is not part of the key: add the suffixes,
                # and place the new field adjacent to the old one
                ndtype[nameidx:nameidx + 1] = [
                    (fname + r1postfix, cdtype),
                    (fname + r2postfix, fdtype)
                ]
    # Rebuild a dtype from the new fields
    ndtype = np.dtype(ndtype)
    # Find the largest nb of common fields :
    # r1cmn and r2cmn should be equal, but...
    cmn = max(r1cmn, r2cmn)
    # Construct an empty array
    output = ma.masked_all((cmn + r1spc + r2spc,), dtype=ndtype)
    names = output.dtype.names
    for f in r1names:
        selected = s1[f]
        if f not in names or (f in r2names and not r2postfix and f not in key):
            f += r1postfix
        current = output[f]
        current[:r1cmn] = selected[:r1cmn]
        if jointype in ('outer', 'leftouter'):
            current[cmn:cmn + r1spc] = selected[r1cmn:]
    for f in r2names:
        selected = s2[f]
        if f not in names or (f in r1names and not r1postfix and f not in key):
            f += r2postfix
        current = output[f]
        current[:r2cmn] = selected[:r2cmn]
        if (jointype == 'outer') and r2spc:
            current[-r2spc:] = selected[r2cmn:]
    # Sort and finalize the output
    output.sort(order=key)
    kwargs = {'usemask': usemask, 'asrecarray': asrecarray}
    return _fix_output(_fix_defaults(output, defaults), **kwargs)


def _rec_join_dispatcher(
        key, r1, r2, jointype=None, r1postfix=None, r2postfix=None,
        defaults=None):
    return (r1, r2)


@array_function_dispatch(_rec_join_dispatcher)
def rec_join(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
             defaults=None):
    """
    Join arrays `r1` and `r2` on keys.
    Alternative to join_by, that always returns a np.recarray.

    See Also
    --------
    join_by : equivalent function
    """
    kwargs = {'jointype': jointype, 'r1postfix': r1postfix, 'r2postfix': r2postfix,
                  'defaults': defaults, 'usemask': False, 'asrecarray': True}
    return join_by(key, r1, r2, **kwargs)


del array_function_dispatch

# === atelier-kyo-manager/app\models.py ===
# app/models.py
from app import db
from datetime import datetime

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    brand = db.Column(db.String(128))
    purchase_price = db.Column(db.Float, nullable=False)
    selling_price = db.Column(db.Float, nullable=False)
    supplier_url = db.Column(db.String(500))
    image_url = db.Column(db.String(500))
    stock_status = db.Column(db.Boolean, default=True)
    profit = db.Column(db.Float)
    listings = db.relationship('Listing', backref='product', lazy='dynamic')

    # 経費項目
    transaction_fee = db.Column(db.Float, default=0.0)
    shipping_cost = db.Column(db.Float, default=0.0)
    customs_duty = db.Column(db.Float, default=0.0)
    procurement_fee = db.Column(db.Float, default=0.0)

    def calculate_profit(self, fee_rate=0.15):
        """None対策済みの利益計算式"""
        calculated_profit = (
            (self.selling_price or 0)
            - (self.purchase_price or 0)
            - ((self.selling_price or 0) * fee_rate)
            - (self.transaction_fee or 0)
            - (self.shipping_cost or 0)
            - (self.customs_duty or 0)
            - (self.procurement_fee or 0)
        )
        self.profit = calculated_profit
        return calculated_profit

    def __repr__(self):
        return f"<Product {self.name}>"

class Listing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(db.String(50), nullable=False)
    fee_rate = db.Column(db.Float)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'))
    listing_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_polynomial_impl.py ===
"""
Functions to operate on polynomials.

"""
__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
           'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
           'polyfit']

import functools
import re
import warnings

import numpy._core.numeric as NX
from numpy._core import (
    abs,
    array,
    atleast_1d,
    dot,
    finfo,
    hstack,
    isscalar,
    ones,
    overrides,
)
from numpy._utils import set_module
from numpy.exceptions import RankWarning
from numpy.lib._function_base_impl import trim_zeros
from numpy.lib._twodim_base_impl import diag, vander
from numpy.lib._type_check_impl import imag, iscomplex, mintypecode, real
from numpy.linalg import eigvals, inv, lstsq

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _poly_dispatcher(seq_of_zeros):
    return seq_of_zeros


@array_function_dispatch(_poly_dispatcher)
def poly(seq_of_zeros):
    """
    Find the coefficients of a polynomial with the given sequence of roots.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Returns the coefficients of the polynomial whose leading coefficient
    is one for the given sequence of zeros (multiple roots must be included
    in the sequence as many times as their multiplicity; see Examples).
    A square matrix (or array, which will be treated as a matrix) can also
    be given, in which case the coefficients of the characteristic polynomial
    of the matrix are returned.

    Parameters
    ----------
    seq_of_zeros : array_like, shape (N,) or (N, N)
        A sequence of polynomial roots, or a square array or matrix object.

    Returns
    -------
    c : ndarray
        1D array of polynomial coefficients from highest to lowest degree:

        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
        where c[0] always equals 1.

    Raises
    ------
    ValueError
        If input is the wrong shape (the input must be a 1-D or square
        2-D array).

    See Also
    --------
    polyval : Compute polynomial values.
    roots : Return the roots of a polynomial.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

    Notes
    -----
    Specifying the roots of a polynomial still leaves one degree of
    freedom, typically represented by an undetermined leading
    coefficient. [1]_ In the case of this function, that coefficient -
    the first one in the returned array - is always taken as one. (If
    for some reason you have one other point, the only automatic way
    presently to leverage that information is to use ``polyfit``.)

    The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`
    matrix **A** is given by

    :math:`p_a(t) = \\mathrm{det}(t\\, \\mathbf{I} - \\mathbf{A})`,

    where **I** is the `n`-by-`n` identity matrix. [2]_

    References
    ----------
    .. [1] M. Sullivan and M. Sullivan, III, "Algebra and Trigonometry,
       Enhanced With Graphing Utilities," Prentice-Hall, pg. 318, 1996.

    .. [2] G. Strang, "Linear Algebra and Its Applications, 2nd Edition,"
       Academic Press, pg. 182, 1980.

    Examples
    --------

    Given a sequence of a polynomial's zeros:

    >>> import numpy as np

    >>> np.poly((0, 0, 0)) # Multiple root example
    array([1., 0., 0., 0.])

    The line above represents z**3 + 0*z**2 + 0*z + 0.

    >>> np.poly((-1./2, 0, 1./2))
    array([ 1.  ,  0.  , -0.25,  0.  ])

    The line above represents z**3 - z/4

    >>> np.poly((np.random.random(1)[0], 0, np.random.random(1)[0]))
    array([ 1.        , -0.77086955,  0.08618131,  0.        ]) # random

    Given a square array object:

    >>> P = np.array([[0, 1./3], [-1./2, 0]])
    >>> np.poly(P)
    array([1.        , 0.        , 0.16666667])

    Note how in all cases the leading coefficient is always 1.

    """
    seq_of_zeros = atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq_of_zeros = eigvals(seq_of_zeros)
    elif len(sh) == 1:
        dt = seq_of_zeros.dtype
        # Let object arrays slip through, e.g. for arbitrary precision
        if dt != object:
            seq_of_zeros = seq_of_zeros.astype(mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(seq_of_zeros) == 0:
        return 1.0
    dt = seq_of_zeros.dtype
    a = ones((1,), dtype=dt)
    for zero in seq_of_zeros:
        a = NX.convolve(a, array([1, -zero], dtype=dt), mode='full')

    if issubclass(a.dtype.type, NX.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = NX.asarray(seq_of_zeros, complex)
        if NX.all(NX.sort(roots) == NX.sort(roots.conjugate())):
            a = a.real.copy()

    return a


def _roots_dispatcher(p):
    return p


@array_function_dispatch(_roots_dispatcher)
def roots(p):
    """
    Return the roots of a polynomial with coefficients given in p.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The values in the rank-1 array `p` are coefficients of a polynomial.
    If the length of `p` is n+1 then the polynomial is described by::

      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    Parameters
    ----------
    p : array_like
        Rank-1 array of polynomial coefficients.

    Returns
    -------
    out : ndarray
        An array containing the roots of the polynomial.

    Raises
    ------
    ValueError
        When `p` cannot be converted to a rank-1 array.

    See also
    --------
    poly : Find the coefficients of a polynomial with a given sequence
           of roots.
    polyval : Compute polynomial values.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

    Notes
    -----
    The algorithm relies on computing the eigenvalues of the
    companion matrix [1]_.

    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    Examples
    --------
    >>> import numpy as np
    >>> coeff = [3.2, 2, 1]
    >>> np.roots(coeff)
    array([-0.3125+0.46351241j, -0.3125-0.46351241j])

    """
    # If input is scalar, this makes it an array
    p = atleast_1d(p)
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    # find non-zero array entries
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return NX.array([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = diag(NX.ones((N - 2,), p.dtype), -1)
        A[0, :] = -p[1:] / p[0]
        roots = eigvals(A)
    else:
        roots = NX.array([])

    # tack any zeros onto the back of the array
    roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots


def _polyint_dispatcher(p, m=None, k=None):
    return (p,)


@array_function_dispatch(_polyint_dispatcher)
def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part

    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}

    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.

    Parameters
    ----------
    p : array_like or poly1d
        Polynomial to integrate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : list of `m` scalars or scalar, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.

        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.

    See Also
    --------
    polyder : derivative of a polynomial
    poly1d.integ : equivalent method

    Examples
    --------

    The defining property of the antiderivative:

    >>> import numpy as np

    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    >>> P
     poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ]) # may vary
    >>> np.polyder(P) == p
    True

    The integration constants default to zero, but can be specified:

    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6,5,3])
    >>> P
    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3. ,  5. ,  3. ]) # may vary

    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:

    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = NX.zeros(m, float)
    k = atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0] * NX.ones(m, float)
    if len(k) < m:
        raise ValueError(
              "k must be a scalar or a rank-1 array of length 1 or >m.")

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        # Note: this must work also with object and integer arrays
        y = NX.concatenate((p.__truediv__(NX.arange(len(p), 0, -1)), [k[0]]))
        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return poly1d(val)
        return val


def _polyder_dispatcher(p, m=None):
    return (p,)


@array_function_dispatch(_polyder_dispatcher)
def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.

    Examples
    --------

    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> import numpy as np

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([0])

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    n = len(p) - 1
    y = p[:-1] * NX.arange(n, 0, -1)
    if m == 0:
        val = p
    else:
        val = polyder(y, m - 1)
    if truepoly:
        val = poly1d(val)
    return val


def _polyfit_dispatcher(x, y, deg, rcond=None, full=None, w=None, cov=None):
    return (x, y, w)


@array_function_dispatch(_polyfit_dispatcher)
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Least squares polynomial fit.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error in the order `deg`, `deg-1`, ... `0`.

    The `Polynomial.fit <numpy.polynomial.polynomial.Polynomial.fit>` class
    method is recommended for new code as it is more stable numerically. See
    the documentation of the method for more information.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (M,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.
    cov : bool or str, optional
        If given and not `False`, return not just the estimate but also its
        covariance matrix. By default, the covariance are scaled by
        chi2/dof, where dof = M - (deg + 1), i.e., the weights are presumed
        to be unreliable except in a relative sense and everything is scaled
        such that the reduced chi2 is unity. This scaling is omitted if
        ``cov='unscaled'``, as is relevant for the case that the weights are
        w = 1/sigma, with sigma known to be a reliable estimate of the
        uncertainty.

    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.

    residuals, rank, singular_values, rcond
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the effective rank of the scaled Vandermonde
           coefficient matrix
        - singular_values -- singular values of the scaled Vandermonde
           coefficient matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    V : ndarray, shape (deg + 1, deg + 1) or (deg + 1, deg + 1, K)
        Present only if ``full == False`` and ``cov == True``.  The covariance
        matrix of the polynomial coefficient estimates.  The diagonal of
        this matrix are the variance estimates for each coefficient.  If y
        is a 2-D array, then the covariance matrix for the `k`-th data set
        are in ``V[:,:,k]``


    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.

        The warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    polyval : Compute polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution minimizes the squared error

    .. math::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2

    in the equations::

        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
        ...
        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]

    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.

    `polyfit` issues a `~exceptions.RankWarning` when the least-squares fit is
    badly conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.

    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           https://en.wikipedia.org/wiki/Polynomial_interpolation

    Examples
    --------
    >>> import numpy as np
    >>> import warnings
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    >>> z
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254]) # may vary

    It is convenient to use `poly1d` objects for dealing with polynomials:

    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179 # may vary
    >>> p(3.5)
    -0.34732142857143039 # may vary
    >>> p(10)
    22.579365079365115 # may vary

    High-order polynomials may oscillate wildly:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', np.exceptions.RankWarning)
    ...     p30 = np.poly1d(np.polyfit(x, y, 30))
    ...
    >>> p30(4)
    -0.80000000000000204 # may vary
    >>> p30(5)
    -0.99999999999999445 # may vary
    >>> p30(4.5)
    -0.10547061179440398 # may vary

    Illustration:

    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    >>> plt.ylim(-2,2)
    (-2, 2)
    >>> plt.show()

    """
    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # set rcond
    if rcond is None:
        rcond = len(x) * finfo(x.dtype).eps

    # set up least squares equation for powers of x
    lhs = vander(x, order)
    rhs = y

    # apply weighting
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs *= w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs *= w[:, NX.newaxis]
        else:
            rhs *= w

    # scale lhs to improve condition number and solve
    scale = NX.sqrt((lhs * lhs).sum(axis=0))
    lhs /= scale
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, resids, rank, s, rcond
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase /= NX.outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError("the number of data points must exceed order "
                                 "to scale the covariance matrix")
            # note, this used to be: fac = resids / (len(x) - order - 2.0)
            # it was decided that the "- 2" (originally justified by "Bayesian
            # uncertainty analysis") is not what the user expects
            # (see gh-11196 and gh-11197)
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return c, Vbase * fac
        else:
            return c, Vbase[:, :, NX.newaxis] * fac
    else:
        return c


def _polyval_dispatcher(p, x):
    return (p, x)


@array_function_dispatch(_polyval_dispatcher)
def polyval(p, x):
    """
    Evaluate a polynomial at specific values.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    If `p` is of length N, this function returns the value::

        p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    If `x` is a sequence, then ``p(x)`` is returned for each element of ``x``.
    If `x` is another polynomial then the composite polynomial ``p(x(t))``
    is returned.

    Parameters
    ----------
    p : array_like or poly1d object
       1D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term, or an
       instance of poly1d.
    x : array_like or poly1d object
       A number, an array of numbers, or an instance of poly1d, at
       which to evaluate `p`.

    Returns
    -------
    values : ndarray or poly1d
       If `x` is a poly1d instance, the result is the composition of the two
       polynomials, i.e., `x` is "substituted" in `p` and the simplified
       result is returned. In addition, the type of `x` - array_like or
       poly1d - governs the type of the output: `x` array_like => `values`
       array_like, `x` a poly1d object => `values` is also.

    See Also
    --------
    poly1d: A polynomial class.

    Notes
    -----
    Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
    for polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.

    If `x` is a subtype of `ndarray` the return value will be of the same type.

    References
    ----------
    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
       Reinhold Co., 1985, pg. 720.

    Examples
    --------
    >>> import numpy as np
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76
    >>> np.polyval([3,0,1], np.poly1d(5))
    poly1d([76])
    >>> np.polyval(np.poly1d([3,0,1]), 5)
    76
    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
    poly1d([76])

    """
    p = NX.asarray(p)
    if isinstance(x, poly1d):
        y = 0
    else:
        x = NX.asanyarray(x)
        y = NX.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y


def _binary_op_dispatcher(a1, a2):
    return (a1, a2)


@array_function_dispatch(_binary_op_dispatcher)
def polyadd(a1, a2):
    """
    Find the sum of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Returns the polynomial resulting from the sum of two input polynomials.
    Each input must be either a poly1d object or a 1D sequence of polynomial
    coefficients, from highest to lowest degree.

    Parameters
    ----------
    a1, a2 : array_like or poly1d object
        Input polynomials.

    Returns
    -------
    out : ndarray or poly1d object
        The sum of the inputs. If either input is a poly1d object, then the
        output is also a poly1d object. Otherwise, it is a 1D array of
        polynomial coefficients from highest to lowest degree.

    See Also
    --------
    poly1d : A one-dimensional polynomial class.
    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval

    Examples
    --------
    >>> import numpy as np
    >>> np.polyadd([1, 2], [9, 5, 4])
    array([9, 6, 6])

    Using poly1d objects:

    >>> p1 = np.poly1d([1, 2])
    >>> p2 = np.poly1d([9, 5, 4])
    >>> print(p1)
    1 x + 2
    >>> print(p2)
       2
    9 x + 5 x + 4
    >>> print(np.polyadd(p1, p2))
       2
    9 x + 6 x + 6

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 + a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) + a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 + NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val


@array_function_dispatch(_binary_op_dispatcher)
def polysub(a1, a2):
    """
    Difference (subtraction) of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Given two polynomials `a1` and `a2`, returns ``a1 - a2``.
    `a1` and `a2` can be either array_like sequences of the polynomials'
    coefficients (including coefficients equal to zero), or `poly1d` objects.

    Parameters
    ----------
    a1, a2 : array_like or poly1d
        Minuend and subtrahend polynomials, respectively.

    Returns
    -------
    out : ndarray or poly1d
        Array or `poly1d` object of the difference polynomial's coefficients.

    See Also
    --------
    polyval, polydiv, polymul, polyadd

    Examples
    --------

    .. math:: (2 x^2 + 10 x - 2) - (3 x^2 + 10 x -4) = (-x^2 + 2)

    >>> import numpy as np

    >>> np.polysub([2, 10, -2], [3, 10, -4])
    array([-1,  0,  2])

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 - a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) - a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 - NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val


@array_function_dispatch(_binary_op_dispatcher)
def polymul(a1, a2):
    """
    Find the product of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Finds the polynomial resulting from the multiplication of the two input
    polynomials. Each input must be either a poly1d object or a 1D sequence
    of polynomial coefficients, from highest to lowest degree.

    Parameters
    ----------
    a1, a2 : array_like or poly1d object
        Input polynomials.

    Returns
    -------
    out : ndarray or poly1d object
        The polynomial resulting from the multiplication of the inputs. If
        either inputs is a poly1d object, then the output is also a poly1d
        object. Otherwise, it is a 1D array of polynomial coefficients from
        highest to lowest degree.

    See Also
    --------
    poly1d : A one-dimensional polynomial class.
    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval
    convolve : Array convolution. Same output as polymul, but has parameter
               for overlap mode.

    Examples
    --------
    >>> import numpy as np
    >>> np.polymul([1, 2, 3], [9, 5, 1])
    array([ 9, 23, 38, 17,  3])

    Using poly1d objects:

    >>> p1 = np.poly1d([1, 2, 3])
    >>> p2 = np.poly1d([9, 5, 1])
    >>> print(p1)
       2
    1 x + 2 x + 3
    >>> print(p2)
       2
    9 x + 5 x + 1
    >>> print(np.polymul(p1, p2))
       4      3      2
    9 x + 23 x + 38 x + 17 x + 3

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1, a2 = poly1d(a1), poly1d(a2)
    val = NX.convolve(a1, a2)
    if truepoly:
        val = poly1d(val)
    return val


def _polydiv_dispatcher(u, v):
    return (u, v)


@array_function_dispatch(_polydiv_dispatcher)
def polydiv(u, v):
    """
    Returns the quotient and remainder of polynomial division.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The input arrays are the coefficients (including any coefficients
    equal to zero) of the "numerator" (dividend) and "denominator"
    (divisor) polynomials, respectively.

    Parameters
    ----------
    u : array_like or poly1d
        Dividend polynomial's coefficients.

    v : array_like or poly1d
        Divisor polynomial's coefficients.

    Returns
    -------
    q : ndarray
        Coefficients, including those equal to zero, of the quotient.
    r : ndarray
        Coefficients, including those equal to zero, of the remainder.

    See Also
    --------
    poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub
    polyval

    Notes
    -----
    Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
    not equal `v.ndim`. In other words, all four possible combinations -
    ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
    ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.

    Examples
    --------

    .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25

    >>> import numpy as np

    >>> x = np.array([3.0, 5.0, 2.0])
    >>> y = np.array([2.0, 1.0])
    >>> np.polydiv(x, y)
    (array([1.5 , 1.75]), array([0.25]))

    """
    truepoly = (isinstance(u, poly1d) or isinstance(v, poly1d))
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k:k + n + 1] -= d * v
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r


_poly_mat = re.compile(r"\*\*([0-9]*)")
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' ' * (len(power) - 1)
        toadd1 = ' ' * (len(partstr) - 1) + power
        if ((len(line2) + len(toadd2) > wrap) or
                (len(line1) + len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' ' * (len(power) - 1)
            line1 += ' ' * (len(partstr) - 1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]


@set_module('numpy')
class poly1d:
    """
    A one-dimensional polynomial class.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    A convenience class, used to encapsulate "natural" operations on
    polynomials so that said operations may take on their customary
    form in code (see Examples).

    Parameters
    ----------
    c_or_r : array_like
        The polynomial's coefficients, in decreasing powers, or if
        the value of the second parameter is True, the polynomial's
        roots (values where the polynomial evaluates to 0).  For example,
        ``poly1d([1, 2, 3])`` returns an object that represents
        :math:`x^2 + 2x + 3`, whereas ``poly1d([1, 2, 3], True)`` returns
        one that represents :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6`.
    r : bool, optional
        If True, `c_or_r` specifies the polynomial's roots; the default
        is False.
    variable : str, optional
        Changes the variable used when printing `p` from `x` to `variable`
        (see Examples).

    Examples
    --------
    >>> import numpy as np

    Construct the polynomial :math:`x^2 + 2x + 3`:

    >>> import numpy as np

    >>> p = np.poly1d([1, 2, 3])
    >>> print(np.poly1d(p))
       2
    1 x + 2 x + 3

    Evaluate the polynomial at :math:`x = 0.5`:

    >>> p(0.5)
    4.25

    Find the roots:

    >>> p.r
    array([-1.+1.41421356j, -1.-1.41421356j])
    >>> p(p.r)
    array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j]) # may vary

    These numbers in the previous line represent (0, 0) to machine precision

    Show the coefficients:

    >>> p.c
    array([1, 2, 3])

    Display the order (the leading zero-coefficients are removed):

    >>> p.order
    2

    Show the coefficient of the k-th power in the polynomial
    (which is equivalent to ``p.c[-(i+1)]``):

    >>> p[1]
    2

    Polynomials can be added, subtracted, multiplied, and divided
    (returns quotient and remainder):

    >>> p * p
    poly1d([ 1,  4, 10, 12,  9])

    >>> (p**3 + 4) / p
    (poly1d([ 1.,  4., 10., 12.,  9.]), poly1d([4.]))

    ``asarray(p)`` gives the coefficient array, so polynomials can be
    used in all functions that accept arrays:

    >>> p**2 # square of polynomial
    poly1d([ 1,  4, 10, 12,  9])

    >>> np.square(p) # square of individual coefficients
    array([1, 4, 9])

    The variable used in the string representation of `p` can be modified,
    using the `variable` parameter:

    >>> p = np.poly1d([1,2,3], variable='z')
    >>> print(p)
       2
    1 z + 2 z + 3

    Construct a polynomial from its roots:

    >>> np.poly1d([1, 2], True)
    poly1d([ 1., -3.,  2.])

    This is the same polynomial as obtained by:

    >>> np.poly1d([1, -1]) * np.poly1d([1, -2])
    poly1d([ 1, -3,  2])

    """
    __hash__ = None

    @property
    def coeffs(self):
        """ The polynomial coefficients """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        # allowing this makes p.coeffs *= 2 legal
        if value is not self._coeffs:
            raise AttributeError("Cannot set attribute")

    @property
    def variable(self):
        """ The name of the polynomial variable """
        return self._variable

    # calculated attributes
    @property
    def order(self):
        """ The order or degree of the polynomial """
        return len(self._coeffs) - 1

    @property
    def roots(self):
        """ The roots of the polynomial, where self(x) == 0 """
        return roots(self._coeffs)

    # our internal _coeffs property need to be backed by __dict__['coeffs'] for
    # scipy to work correctly.
    @property
    def _coeffs(self):
        return self.__dict__['coeffs']

    @_coeffs.setter
    def _coeffs(self, coeffs):
        self.__dict__['coeffs'] = coeffs

    # alias attributes
    r = roots
    c = coef = coefficients = coeffs
    o = order

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._variable = c_or_r._variable
            self._coeffs = c_or_r._coeffs

            if set(c_or_r.__dict__) - set(self.__dict__):
                msg = ("In the future extra properties will not be copied "
                       "across when constructing one poly1d from another")
                warnings.warn(msg, FutureWarning, stacklevel=2)
                self.__dict__.update(c_or_r.__dict__)

            if variable is not None:
                self._variable = variable
            return
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = atleast_1d(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError("Polynomial must be 1d only.")
        c_or_r = trim_zeros(c_or_r, trim='f')
        if len(c_or_r) == 0:
            c_or_r = NX.array([0], dtype=c_or_r.dtype)
        self._coeffs = c_or_r
        if variable is None:
            variable = 'x'
        self._variable = variable

    def __array__(self, t=None, copy=None):
        if t:
            return NX.asarray(self.coeffs, t, copy=copy)
        else:
            return NX.asarray(self.coeffs, copy=copy)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return f"poly1d({vals})"

    def __len__(self):
        return self.order

    def __str__(self):
        thestr = "0"
        var = self.variable

        # Remove leading zeros
        coeffs = self.coeffs[NX.logical_or.accumulate(self.coeffs != 0)]
        N = len(coeffs) - 1

        def fmt_float(q):
            s = f'{q:.4g}'
            s = s.removesuffix('.0000')
            return s

        for k, coeff in enumerate(coeffs):
            if not iscomplex(coeff):
                coefstr = fmt_float(real(coeff))
            elif real(coeff) == 0:
                coefstr = f'{fmt_float(imag(coeff))}j'
            else:
                coefstr = f'({fmt_float(real(coeff))} + {fmt_float(imag(coeff))}j)'

            power = (N - k)
            if power == 0:
                if coefstr != '0':
                    newstr = f'{coefstr}'
                elif k == 0:
                    newstr = '0'
                else:
                    newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = f'{coefstr} {var}'
            elif coefstr == '0':
                newstr = ''
            elif coefstr == 'b':
                newstr = '%s**%d' % (var, power,)
            else:
                newstr = '%s %s**%d' % (coefstr, var, power)

            if k > 0:
                if newstr != '':
                    if newstr.startswith('-'):
                        thestr = f"{thestr} - {newstr[1:]}"
                    else:
                        thestr = f"{thestr} + {newstr}"
            else:
                thestr = newstr
        return _raise_power(thestr)

    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError("Power to non-negative integers only.")
        res = [1]
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __truediv__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs / other)
        else:
            other = poly1d(other)
            return polydiv(self, other)

    def __rtruediv__(self, other):
        if isscalar(other):
            return poly1d(other / self.coeffs)
        else:
            other = poly1d(other)
            return polydiv(other, self)

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return (self.coeffs == other.coeffs).all()

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        return not self.__eq__(other)

    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order:
            return self.coeffs.dtype.type(0)
        if val < 0:
            return self.coeffs.dtype.type(0)
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            zr = NX.zeros(key - self.order, self.coeffs.dtype)
            self._coeffs = NX.concatenate((zr, self.coeffs))
            ind = 0
        self._coeffs[ind] = val

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        """
        Return an antiderivative (indefinite integral) of this polynomial.

        Refer to `polyint` for full documentation.

        See Also
        --------
        polyint : equivalent function

        """
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `polyder` for full documentation.

        See Also
        --------
        polyder : equivalent function

        """
        return poly1d(polyder(self.coeffs, m=m))

# Stuff to do on module import


warnings.simplefilter('always', RankWarning)