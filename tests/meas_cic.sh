#!/bin/bash
# 
# @file meas_cic.sh 
# @author m. s. sūryan śivadās
#


# status variables
ERROR=1; WARNING=2; INFO=0

# python executable name or path
PYTHON=python3 

# mpi arguments
MPI_ARGS="" # "mpiexec -np 16"

# logging function
log(){

    if [ $2 -eq  $ERROR ]; then

        # error
        # echo -e "$(date +"%F %T") \033[1m\033[91m$1 \033[0m"
        echo -e "\033[1m\033[91m$1 \033[0m"

    elif [ $2 -eq $WARNING ]; then

        # warning
        echo -e "\033[1m\033[1m\033[93m$1\033[0m"

    else
        
        # info
        echo -e "\033[1m\033[92m$1\033[0m"

    fi
}



log "Using $(${PYTHON} --version)" $INFO 


PROG=/home/ms3/Documents/phd/cosmo/codes/cosmology_codes/count_in_cells/app.py
if [ ! -e $PROG ]; then

    log "Cannot find program: ${PROG}, exiting :(" $ERROR
    exit 1

fi 


# zenity --notification --text='Starting measuring count-in-cells job!...'
log 'Starting measuring count-in-cells job!...' $INFO

OPTFILES=(param.yml param1.yml)
for OPTFILE in ${OPTFILES[@]}; do 
    
    # check if the file exist, if yes, run the scrip
    if [ ! -e $OPTFILE ]; then
        log "File does not exist: ${OPTFILE}, skipping... :(" $WARNING
    else

        log "echo Using options file: ${OPTFILE}" $INFO

        # run the program with given options file
        log "Executing '${MPI_ARGS} ${PYTHON} ${PROG} --opt-file=${OPTFILE}'" $INFO
        ${MPI_ARGS} ${PYTHON} ${PROG} --opt-file=${OPTFILE}

    fi

done

# zenity --notification --text='Finished measuring count-in-cells job! :)'
log 'Finished measuring count-in-cells job! :)' $INFO
