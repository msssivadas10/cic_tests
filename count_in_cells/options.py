#!/usr/bin/python3

import yaml, json # for loading the parameters file in yaml/json format
from dataclasses import dataclass, asdict
from collections import namedtuple
from utils import WARN, ERROR, SUCCESS



@dataclass(slots = True, frozen = True, kw_only = True)
class Options:

    catalog_chunk_size: int
    catalog_compression: str
    catalog_magnitude: str
    catalog_magnitude_offset: str
    catalog_mask: str
    catalog_object: str
    catalog_random: str
    catalog_redshift: str
    catalog_redshift_error: str
    catalog_all_bands: list
    catalog_dec_shift: float
    catalog_ra_shift: float 
    catalog_object_filter_conditions: list
    catalog_random_filter_conditions: list
    catalog_magnitude_to_correct: list
    cic_cell_num_subdiv: int
    cic_cellsize: float
    cic_redshift_filter_conditions: list
    cic_magnitude_filter_conditions: float
    cic_use_mask: list
    cic_max_count: int 
    cic_masked_frac: float
    cic_save_counts: bool
    jackknife_patch_width_ra: float
    jackknife_patch_width_dec: float
    jackknife_region_rect: list
    jackknife_remove_regions: list
    jackknife_use_mask: list
    output_dir: str

    def save_as(self, file: str):

        with open( file, 'w' ) as fp:
            for __key, __value in asdict( self ).items():
                fp.write( f"{__key:32s} = {__value}\n" )
        return 
        

_OptionBlock = namedtuple( '_OptionBlock', ['name', 'fields', 'optional', 'value'], defaults = [False, None] )
_OptionField = namedtuple( '_OptionField', ['name', 'optional', 'value'], defaults = [False, None] )
_Message     = namedtuple( '_Message', ['msg', 'status'], defaults = [ERROR] )

# structure of the options
opt_tree = [
            _OptionBlock( 
                            name   = 'catalog',
                            fields = [
                                        _OptionField( 'object' ),
                                        _OptionField( 'random' ),
                                        _OptionField( 'compression',              optional = True, value = 'infer'   ),
                                        _OptionField( 'chunk_size',               optional = True, value = 1_000_000 ),
                                        _OptionField( 'ra_shift',                 optional = True, value = 0.0       ),
                                        _OptionField( 'dec_shift',                optional = True, value = 0.0       ),
                                        _OptionField( 'redshift',                 optional = True, value = "redshift"           ),
                                        _OptionField( 'redshift_error',           optional = True, value = "redshift_error"     ),
                                        _OptionField( 'magnitude',                optional = True, value = "%(band)_mag"        ),
                                        _OptionField( 'magnitude_offset',         optional = True, value = "%(band)_mag_offset" ), 
                                        _OptionField( 'mask',                     optional = True, value = "%(band)_mask"       ),
                                        _OptionField( 'all_bands',                optional = True, value = ['g','r','i','z','y']),
                                        _OptionField( 'object_filter_conditions', optional = True, value = []    ),
                                        _OptionField( 'random_filter_conditions', optional = True, value = []    ),
                                        _OptionField( 'magnitude_to_correct',     optional = True, value = []    ),
                                    ] 
                        ),
            _OptionBlock( 
                            name = 'cic',
                            fields = [
                                        _OptionField( 'cellsize'  ),
                                        _OptionField( 'max_count' ),
                                        _OptionField( 'use_mask'  ),
                                        _OptionField( 'cell_num_subdiv',             optional = True, value = 0  ),
                                        _OptionField( 'redshift_filter_conditions',  optional = True, value = [] ),
                                        _OptionField( 'magnitude_filter_conditions', optional = True, value = [] ),
                                        _OptionField( 'save_counts',                 optional = True, value = False ),
                                        _OptionField( 'masked_frac',                 optional = True, value = 0.05  ),
                                     ] 
                        ),
            _OptionBlock( 
                            name = 'jackknife',
                            fields = [
                                        _OptionField( 'region_rect'     ),
                                        _OptionField( 'patch_width_ra'  ),
                                        _OptionField( 'patch_width_dec' ),
                                        _OptionField( 'use_mask'        ),
                                        _OptionField( 'remove_regions', optional = True, value = [] ),
                                     ]
                        ),
            _OptionField( 'output_dir', optional = True, value = "./output" )
        ]


    
def load_options(file: str):
    r"""
    Load options from a YAML/JSON file and run a check on options.
    """

    def __load_options(fp):

        # try yaml format first
        try:
            return yaml.safe_load( fp )
        except Exception:
            pass

        # try json format then
        try:
            return json.load( fp )
        except Exception:
            pass

        # raise error: not a valid json or yaml
        raise ValueError(f"{file} must be a valid JSON or YAML file")


    with open(file, 'r') as fp:
        _opts = __load_options( fp )

    msgs, opts = [], {} 
    for item in opt_tree:

        # if the item is a field
        if isinstance( item, _OptionField ):
            
            field_value = _opts.get( item.name )
            if field_value is None:
                field_value = item.value
                msgs.append( _Message( f"setting optional field '{item.name}' to {field_value}", WARN ) )
            opts[ item.name ] = field_value
            continue

        # if the item is a block
        block_value = _opts.get( item.name )
        if block_value is None:
            
            # if not optional block, set error and all field values to None
            if not item.optional:
                msgs.append( _Message( f"missing required block: '{ item.name }'", ERROR ) )

                for field in item.fields:
                    opts[ '_'.join([ item.name, field.name ]) ] = field.value
                continue
            
            block_value = item.value # if optional set the default value

        for field in item.fields:

            field_value = block_value.get( field.name )
            if field_value is None:

                if not field.optional:
                    msgs.append( _Message( f"missing required field: '{ field.name }'", ERROR ) )
                    opts[ '_'.join([ item.name, field.name ]) ] = None
                    continue
                
                field_value = field.value
                msgs.append( _Message( f"setting optional field '{item.name}_{field.name}' to {field_value}", WARN ) )

            opts[ '_'.join([ item.name, field.name ]) ] = field_value
    
    msgs = sorted( msgs, key = lambda __msg: __msg.status )
    return Options( **opts ), msgs




