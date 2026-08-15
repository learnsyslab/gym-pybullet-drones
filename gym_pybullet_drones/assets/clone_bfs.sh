#!/bin/bash

# Clone, edit, build, configure, multiple Betaflight SITL executables

# Check for the correct number of command-line arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <desired_max_num_drones>"
    exit 1
fi

# Extract command-line arguments
desired_max_num_drones="$1"

# Create gitignored directory in gym-pybullet-donres
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ../../
mkdir betaflight_sitl/
cd betaflight_sitl/

# Step 1: Clone and open betaflight's source (at the time of writing, branch `master`, future release 4.5)):
git clone https://github.com/betaflight/betaflight temp/


# Step 2: Comment out line `delayMicroseconds_real(50); // max rate 20kHz` 
# (https://github.com/betaflight/betaflight/blob/master/src/main/main.c#L52) 
# from Betaflight's `SIMULATOR_BUILD`
cd temp/

git checkout cafe727 #latest commit at the time of writing the gym-pybullet-drones Readme

sed -i "s/delayMicroseconds_real(50);/\/\/delayMicroseconds_real(50);/g" ./src/main/main.c
sed -i "s/\"-ABEFR\"/{ '-', 'A', 'B', 'E', 'F', 'R' }/g" ./src/main/drivers/vtx_table.c

# Prepare
make arm_sdk_install 

cd ..

pattern1="PORT_PWM_RAW    9001"
pattern2="PORT_PWM        9002"
pattern3="PORT_STATE      9003"
pattern4="PORT_RC         9004"
# pattern5="BASE_PORT 5760"

for ((i = 0; i < desired_max_num_drones; i++)); do

    # Copy
    cp -r temp/ "bf${i}/"
    cd "bf${i}/"

    # Step 3: Change the UDP ports used by each Betaflight SITL instance
    replacement1="PORT_PWM_RAW    90${i}1"
    sed -i "s/$pattern1/$replacement1/g" ./src/main/target/SITL/sitl.c
    replacement2="PORT_PWM    90${i}2"
    sed -i "s/$pattern2/$replacement2/g" ./src/main/target/SITL/sitl.c
    replacement3="PORT_STATE    90${i}3"
    sed -i "s/$pattern3/$replacement3/g" ./src/main/target/SITL/sitl.c
    replacement4="PORT_RC    90${i}4"
    sed -i "s/$pattern4/$replacement4/g" ./src/main/target/SITL/sitl.c
    # replacement5="BASE_PORT 57${i}0"
    # sed -i "s/$pattern5/$replacement5/g" ./src/main/drivers/serial_tcp.c

    # Build
    make TARGET=SITL

    cd ..

done

for ((i = 0; i < desired_max_num_drones; i++)); do

    # Step 4: Copy over the configured `eeprom.bin` file from folder 
    cp "${SCRIPT_DIR}/eeprom.bin" "bf${i}/"

done
