#!/usr/bin/python3

import yaml, json # for loading the parameters file in yaml/json format
from dataclasses import dataclass, asdict
from collections import namedtuple
from utils import WARN, ERROR, SUCCESS


# options for count-in-cells calculation
@dataclass(slots = True, frozen = True, kw_only = True)
class Options:
    catalog_chunk_size               : int   = None
    catalog_compression              : str   = None
    catalog_magnitude                : str   = None
    catalog_magnitude_offset         : str   = None
    catalog_mask                     : str   = None
    catalog_object                   : str   = None
    catalog_random                   : str   = None
    catalog_redshift                 : str   = None
    catalog_redshift_error           : str   = None
    catalog_all_bands                : list  = None
    catalog_object_filter_conditions : list  = None
    catalog_random_filter_conditions : list  = None
    catalog_magnitude_to_correct     : list  = None
    cic_cell_num_subdiv              : int   = None
    cic_cellsize                     : float = None
    cic_redshift_filter_conditions   : list  = None
    cic_magnitude_filter_conditions  : float = None
    cic_use_mask                     : list  = None
    cic_max_count                    : int   = None 
    cic_masked_frac                  : float = None
    cic_save_counts                  : bool  = None
    cic_do_stats                     : bool  = None
    jackknife_patch_width_ra         : float = None
    jackknife_patch_width_dec        : float = None
    jackknife_region_rect            : list  = None
    jackknife_remove_regions         : list  = None
    jackknife_use_mask               : list  = None
    cumstats_data_files              : list  = None    
    cumstats_max_count               : int   = None   
    cumstats_cell_num_subdiv         : int   = None         
    cumstats_masked_frac             : float = None     
    output_dir                       : str   = None

    def save_as(self, file: str):

        with open( file, 'w' ) as fp:
            for __key, __value in asdict( self ).items():
                if __value is None:
                    continue
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
                                        _OptionField( 'cell_num_subdiv',             optional = True, value = 0    ),
                                        _OptionField( 'redshift_filter_conditions',  optional = True, value = []   ),
                                        _OptionField( 'magnitude_filter_conditions', optional = True, value = []   ),
                                        _OptionField( 'save_counts',                 optional = True, value = True ),
                                        _OptionField( 'do_stats',                    optional = True, value = True ),
                                        _OptionField( 'masked_frac',                 optional = True, value = 0.05 ),
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
            _OptionBlock(
                            name   = 'cumstats',
                            fields = [
                                        _OptionField( 'data_files' ),
                                        _OptionField( 'max_count'  ),
                                        _OptionField( 'cell_num_subdiv', optional = True, value = 0    ),
                                        _OptionField( 'masked_frac',     optional = True, value = 0.05 ),
                                     ]
                        ),
            _OptionField( 'output_dir', optional = True, value = "./output" ),
        ]


def __load_options(file: str):

    # try yaml format first
    try:
        with open(file, 'r') as fp:
            return yaml.safe_load( fp )
    except Exception:
        pass

    # try json format then
    try:
        with open(file, 'r') as fp:
            return json.load( fp )
    except Exception:
        pass

    # raise error: not a valid json or yaml
    raise ValueError(f"{file} must be a valid JSON or YAML file")

    
def load_options(file: str, base_file: str = None, task_code: int = 0):
    r"""
    Load count-in-cells measurements options from a YAML/JSON file.
    """

    # set unwanted items of the options file, based on the task 
    unwanted_items = []
    if task_code == 0: # count estimation with optional distribution calculation
        unwanted_items = ['cumstats']
    elif task_code == 1: # combining regional results to calculate distribution
        unwanted_items = ['catalog', 'cic', 'jackknife']

    _opts = __load_options( file )
    
    # base tree from which missing values are inherited
    _base = {} if base_file is None else __load_options( base_file )

    msgs, opts = [], {} 
    error_code = SUCCESS
    for item in opt_tree:

        # ignore if the item is unwanted in the present context
        if item.name in unwanted_items:
            continue

        # if the item is a field
        if isinstance( item, _OptionField ):
            
            field_value      = _opts.get( item.name )
            base_field_value = _base.get( item.name ) 
            if field_value is None:

                if base_field_value is None:

                    if item.optional:
                        field_value = item.value
                        msgs.append( _Message( f"setting optional field '{item.name}' to {field_value}", WARN ) )
                    else:
                        field_value = None
                        error_code  = ERROR
                        msgs.append( _Message( f"missing required field '{item.name}'", ERROR ) )
                else:
                    field_value = base_field_value
                    msgs.append( _Message( f"inheriting value of field '{item.name}' as {field_value}", WARN ) )

            opts[ item.name ] = field_value
            continue


        # if the item is a block
        block_value      = _opts.get( item.name )
        base_block_value = _base.get( item.name )
        if block_value is None:

            # if not optional block,
            if not item.optional:

                # if base value is given, use it, else set error and all field values to None
                if base_block_value is None:

                    msgs.append( _Message( f"missing required block: '{ item.name }'", ERROR ) )
                    error_code = ERROR

                    for field in item.fields:
                        opts[ '_'.join([ item.name, field.name ]) ] = field.value
                    continue
                else:
                    msgs.append( _Message( f"inheriting value of required block: '{ item.name }'", WARN ) )
                    block_value = base_block_value
            else:
                block_value = item.value # if optional set the default value

        # in case no base file given...
        base_block_value = {} if base_block_value is None else base_block_value

        # load field values
        for field in item.fields:

            field_value      = block_value.get( field.name )
            base_field_value = base_block_value.get( field.name )
            if field_value is None:

                if not field.optional:

                    if base_field_value is None:

                        msgs.append( _Message( f"missing required field: '{ item.name }.{ field.name }'", ERROR ) )
                        error_code = ERROR
                        opts[ '_'.join([ item.name, field.name ]) ] = None
                        continue
                    else:
                        field_value = base_field_value
                        msgs.append( _Message( f"inheriting value of field '{item.name}.{ field.name }' as {field_value}", WARN ) )
                else:
                    field_value = field.value
                    msgs.append( _Message( f"setting optional field '{item.name}.{field.name}' to {field_value}", WARN ) )

            opts[ '_'.join([ item.name, field.name ]) ] = field_value
    
    return Options( **opts ), msgs, error_code

