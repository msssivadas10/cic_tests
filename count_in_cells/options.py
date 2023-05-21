#!/usr/bin/python3

import os
import json # for loading the parameters file in json format
from dataclasses import dataclass
from typing import Any, Callable, Sequence

#############################################################################################
# CHECKING UTILITIES
#############################################################################################

WARN, ERROR = 2, 1
INT, FLOAT, STR, LEFT, RIGHT = 1, 2, 4, 1, 2
NUMERIC = INT | FLOAT 
ANY     = NUMERIC | STR


@dataclass(slots = True)
class _CheckerResult:

    value: Any   = None   # 
    error: str   = None   # error messages
    warnings: list = None # warning messages

    def add_message(self, message: str, level: int = ERROR) -> int:

        if level == ERROR:
            self.error = message
            return 1
        
        # level == WARN
        if self.warnings is None:
            self.warnings = []
        self.warnings.append( message )
        return 0
    
    @property
    def flag(self) -> int:

        # failure flag. 2 = warning, 1 = failure, 0 = success
        if self.error is not None:
            return ERROR
        if self.warnings is not None:
            return WARN
        return 0


def check_path(path: str, level: int = ERROR):

    res = _CheckerResult()
    if not os.path.exists(path):
        res.add_message( f"{ path } is not a valid path", level )
    return res


def check_state(x: Any, values: tuple = None, level: int = ERROR, allow_none: bool = False):

    res = _CheckerResult()
    if allow_none and x is None:
        return res

    if not isinstance(x, (str, int)):
        res.add_message( "value must be a string or integer", ERROR )
        return res

    if values is not None:
        if all( map( lambda _y: x != _y, values ) ):
            msg = "incorrect value: '{}'; possible value(s): {}".format( x, ', '.join( map( str, values ) ) )
            if res.add_message( msg, level ):
                return res
    
    return res


def check_basetype(x: Any, type: int = ANY):

    res = _CheckerResult()

    types = {}
    if type & INT:
        types['int'] = int
    if type & FLOAT:
        types['float'] = float
    if type & STR:
        types['str'] = str

    if not isinstance(x, tuple( types.values() )):
        if res.add_message( "incorrect type: must be {}".format( ' or '.join( types.keys() ) ), ERROR ):
            return res
    return res


def check_sequence(x: Sequence, type: int = ANY, size: int = None):

    res = _CheckerResult()

    try:
        _size = len(x)
        if size is not None:
            if _size != size:
                res.add_message( f"sequence should have size {size}, got {_size}", ERROR )
                return res 
    except Exception:
        res.add_message( "value is not a sequence" )

    for value in x:
        res2 = check_basetype( value, type )
        if res2.flag == ERROR:
            res.add_message( f"sequence contain { res2.error }", ERROR )
    
    return res


def check_number(x: Any, type: int = NUMERIC, level: int = ERROR, *, value: Any = None, 
                 lower: Any = None, upper: Any = None, closed: int = 3, allow_none: bool = False):
    
    types = {}
    if type & INT:
        types['int'] = int
    if type & FLOAT:
        types['float'] = float

    res = _CheckerResult()
    if allow_none and x is None:
        return res
    
    if not isinstance(x, tuple( types.values() )):
        if res.add_message( "incorrect type: must be {}".format( ' or '.join( types.keys() ) ), ERROR ):
            return res
    
    if value is not None:
        if x != value:
            res.add_message( f"incorrect value: must be {value}", level )
        return res
    
    if lower is not None:
        if closed & LEFT:
            if x < lower:
                if res.add_message( f"value must be greater than or equal to {lower}, got {x}", level ):
                    return res
        elif x <= lower:
            if res.add_message( f"value must be greater than {lower}, got {x}", level ):
                return res
        
    if upper is not None:
        if closed & RIGHT:
            if x > upper:
                if res.add_message( f"value must be less than or equal to {upper}, got {x}", level ):
                    return res
        elif x >= upper:
            if res.add_message( f"value must be less than {upper}, got {x}", level ):
                return res
        
    return res



##############################################################################################
# LOADING OPTIONS + CHECKING
##############################################################################################


@dataclass(slots = True, frozen = True, kw_only = True)
class Options:

    catalog_chunk_size: int
    catalog_compression: str
    catalog_magnitude: str
    catalog_magnitude_offset: str
    catalog_mask: str
    catalog_object: str
    catalog_random: str
    catalog_z_prefix: str
    catalog_z_suffix: str
    catalog_filters: list
    catalog_dec_shift: float
    catalog_ra_shift: float 

    cic_cell_num_subdiv: int
    cic_cellsize_arcsec: float
    cic_mag_upper_cutoff: dict
    cic_redshift: float
    cic_redshift_width: float
    cic_redshift_filter: list
    cic_use_mask: list
    cic_max_count: int 


    jackknife_patch_xwidth: float
    jackknife_patch_ywidth: float
    jackknife_region_rect: list
    jackknife_remove_regions: list
    jackknife_use_mask: list

    output_path: str
    output_patch_img_dir: str


@dataclass( slots = True, frozen = True )
class _OptionBlock:
    name: str
    fields: list
    optional: bool    = False
    value: Any        = None


@dataclass( slots = True, frozen = True )
class _OptionField:
    name: str
    checker: Callable = None
    optional: bool    = False
    value: Any        = None


# options tree
opt_tree = [
            _OptionBlock( 
                            name   = 'catalog',
                            fields = [
                                        _OptionField( 
                                                        name = 'object', 
                                                        checker = check_path 
                                                    ),
                                        _OptionField( 
                                                        name = 'random', 
                                                        checker = check_path 
                                                    ),
                                        _OptionField(
                                                        name = 'compression',
                                                        checker = lambda x: check_state(x, allow_none = True),
                                                        optional = True,
                                                        value = None
                                                    ),
                                        _OptionField(
                                                        name = 'chunk_size',
                                                        checker = lambda x: check_number(x, type = INT, lower = 1000),
                                                        optional = True,
                                                        value = 1_000_000
                                                    ),
                                        _OptionField(
                                                        name = 'ra_shift',
                                                        checker = check_number,
                                                        optional = True,
                                                        value = 0.0
                                                    ),
                                        _OptionField(
                                                        name = 'dec_shift',
                                                        checker = check_number,
                                                        optional = True,
                                                        value = 0.0
                                                    ),
                                        _OptionField(
                                                        name = 'z_prefix',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = "redshift"
                                                    ),
                                        _OptionField(
                                                        name = 'z_suffix',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = ""
                                                    ),
                                        _OptionField(
                                                        name = 'magnitude',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = "{}_mag"
                                                    ),
                                        _OptionField(
                                                        name = 'magnitude_offset',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = "{}_mag_offset"
                                                    ),
                                        _OptionField(
                                                        name = 'mask',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = "{}_mask"
                                                    ),
                                        _OptionField(
                                                        name = 'filters',
                                                        checker = lambda x: check_sequence(x, STR),
                                                        optional = True,
                                                        value = []
                                                    ),
                                    ] 
                        ),
            _OptionBlock( 
                            name = 'cic',
                            fields = [
                                        _OptionField(
                                                        name = 'cellsize_arcsec',
                                                        checker = lambda x: check_number(x, NUMERIC, lower = 0.)
                                                    ),
                                        _OptionField(
                                                        name = 'cell_num_subdiv',
                                                        checker = lambda x: check_number(x, INT, lower = 1, upper = 100),
                                                        optional = True,
                                                        value = 1
                                                    ),
                                        _OptionField(
                                                        name = 'mag_upper_cutoff',
                                                        checker = None, # dict, processed later
                                                        optional = True,
                                                        value = {}
                                                    ),
                                        _OptionField(
                                                        name = 'use_mask',
                                                        checker = lambda x: check_sequence(x, STR)
                                                    ),
                                        _OptionField(
                                                        name = 'redshift',
                                                        checker = lambda x: check_number(x, NUMERIC, lower = -1., closed = 0)
                                                    ),
                                        _OptionField(
                                                        name = 'redshift_width',
                                                        checker = lambda x: check_number(x, NUMERIC, lower = 0.)
                                                    ),
                                        _OptionField(
                                                        name = 'redshift_filter',
                                                        checker = lambda x: check_sequence(x, STR),
                                                        optional = True,
                                                        value = []
                                                    ),
                                        _OptionField(
                                                        name = 'max_count',
                                                        checker = lambda x: check_number(x, INT, lower = 10)
                                                    ),
                                     ] 
                        ),
            _OptionBlock( 
                            name = 'jackknife',
                            fields = [
                                        _OptionField(
                                                        name = 'region_rect',
                                                        checker = lambda x: check_sequence(x, NUMERIC, size = 4)
                                                    ),
                                        _OptionField(
                                                        name = 'remove_regions',
                                                        checker = None, # TODO: define checker
                                                        optional = True,
                                                        value = []
                                                    ),
                                        _OptionField(
                                                        name = 'patch_xwidth',
                                                        checker = lambda x: check_number(x, NUMERIC, lower = 0., upper = 360., closed = 0)
                                                    ),
                                        _OptionField(
                                                        name = 'patch_ywidth',
                                                        checker = lambda x: check_number(x, NUMERIC, lower = 0., upper = 360., closed = 0)
                                                    ),
                                        _OptionField(
                                                        name = 'use_mask',
                                                        checker = lambda x: check_sequence(x, STR)
                                                    ),
                                     ]
                        ),
            _OptionBlock( 
                            name = 'output',
                            fields = [
                                        _OptionField(
                                                        name = 'path',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = "./output"
                                                    ),
                                        _OptionField(
                                                        name = 'patch_img_dir',
                                                        checker = lambda x: check_basetype(x, STR),
                                                        optional = True,
                                                        value = "patch_images"
                                                    ),
                                     ], 
                            optional = True, 
                            value = {} 
                        )
        ]


    
def load_options(file: str):
    r"""
    Load options from a JSON file and run a check on options.
    """

    with open(file, 'r') as fp:
        _opts = json.load(fp)


    # TODO: options check

    res  = _CheckerResult()
    opts = {} 
    for block in opt_tree:
        
        blk_value = _opts.get( block.name )
        if blk_value is None:
            
            if not block.optional:
                res.add_message( f"missing required block: '{ block.name }'" )
                return res
            
            blk_value = block.value

        for field in block.fields:

            field_value = blk_value.get( field.name )
            if field_value is None:

                if not field.optional:
                    res.add_message( f"missing required field: '{ field.name }'" )
                    return res
                
                field_value = field.value

            check_res = _CheckerResult()
            if field.checker:
                check_res = field.checker( field_value )
            if check_res.flag:
                res.add_message( f"incorrect value for field '{ block.name }.{ field.name }': { check_res.error }" )
                return res
            
            opts[ '_'.join([ block.name, field.name ]) ] = field_value
    

    res.value = Options( **opts )
    return res



if __name__ == '__main__':

    import pprint
    file = os.path.join(os.path.split(__file__)[0], "param.json")
    f = load_options(file) 
    pprint.pprint( f )
    # print( f.message )
