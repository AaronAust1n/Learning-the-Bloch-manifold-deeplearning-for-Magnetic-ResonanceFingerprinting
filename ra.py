import struct
import numpy as np

FLAG_BIG_ENDIAN = 0b1
FLAG_COMPRESSED = 0b10
MAGIC_NUMBER = 8746397786917265778
dtype_kind_to_enum = {'i':1,'u':2,'f':3,'c':4}
dtype_enum_to_name = {0:'user',1:'int',2:'uint',3:'float',4:'complex'}

def read(filename):
    f = open(filename,'rb')
    h = getheader(f)
    h['dims'] = h['dims'][::-1]
    data = f.read()
    if h['eltype'] == 0:
        print('Unable to convert user data. Returning raw byte string.')
    else:
        d = '%s%d' % (dtype_enum_to_name[h['eltype']], h['elbyte']*8)
        data = np.fromstring(data, dtype=np.dtype(d))
        data = data.reshape(h['dims']).transpose()
    f.close()
    return data


def getheader(f):
    buf = f.read(48)
    header_data = np.frombuffer(buf, '<Q')
    h = dict()
    h['flags'] = header_data[1]
    h['eltype'] = header_data[2]
    h['elbyte'] = header_data[3]
    h['size'] = header_data[4]
    h['ndims'] = header_data[5]
    buf = f.read(int(8*h['ndims']))
    h['dims'] = np.frombuffer(buf, '<Q')
    return h


def query(filename):
    q = "---\nname: %s\n" % filename
    fd = open(filename,'r')
    h = getheader(fd)
    fd.close()
    if h['flags'] & FLAG_BIG_ENDIAN != 0:
        endian = 'big'
    else:
        endian = 'little'
    assert endian == 'little'  # big not implemented yet
    q += 'endian: %s\n' % endian
    q += 'type: %s%d\n' % (dtype_enum_to_name[h['eltype']], h['elbyte']*8)
    q += 'size: %d\n' % h['size']
    q += 'dimension: %d\n' % h['ndims']
    q += 'shape:\n'
    for d in h['dims']:
        q += '  - %d\n' % d
    q += '...'
    return q


def write(data, filename):
    flags = 0
    if data.dtype.str[0] == '>':
        flags |= FLAG_BIG_ENDIAN
    try:
        eltype = dtype_kind_to_enum[data.dtype.kind]
    except KeyError:
        eltype = 0
    elbyte = data.dtype.itemsize
    size = data.size*elbyte
    ndims = len(data.shape)
    dims = data.shape
    dims = np.array([_ for _ in data.shape]).astype('uint64')
    f = open(filename,'wb')
    f.write(struct.pack('<Q', MAGIC_NUMBER))
    f.write(struct.pack('<Q', flags))
    f.write(struct.pack('<Q', eltype))
    f.write(struct.pack('<Q', elbyte))
    f.write(struct.pack('<Q', size))
    f.write(struct.pack('<Q', ndims))
    f.write(dims.tobytes())
    f.write(data.transpose().tobytes())
    f.close()
