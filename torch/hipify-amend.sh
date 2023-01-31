# do not use map because the keys are unordered
substitute_str=("<ATen\/cuda\/HIPContext.h>=<ATen\/hip\/HIPContext.h>"
                "<c10\/cuda\/CUDAGuard.h>=<c10\/hip\/HIPGuard.h>"
                "<c10\/cuda\/CUDAStream.h>=<c10\/hip\/HIPStream.h>"
                "<THC\/THCAtomics.cuh>=<THH\/THHAtomics.cuh>"
                "<hipDNN.h>=<miopen\/miopen.h>"
                "CUDAGuard=HIPGuard"
                "CUDAStream=HIPStream"
                "cudnn=miopen"
                'namespace cuda=namespace hip'
)

file_dir=`dirname $0`
echo $file_dir

for (( i = 0; i < ${#substitute_str[@]}; i++ ));do
    OLD_IFS=${IFS}
    IFS=$'='
    a=(${substitute_str[$i]})
    old_str=${a[0]}
    new_str=${a[1]}
    IFS=${OLD_IFS}

    echo $old_str
    echo $new_str
    echo "s/${old_str}/${new_str}/g"
    sed -i "s/${old_str}/${new_str}/g" `grep -rl "${old_str}" $file_dir --exclude=*.sh`
done
