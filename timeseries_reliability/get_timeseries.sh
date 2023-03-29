#!/usr/bin/env bash


subjects='sub-DP5001  sub-DP5004    sub-DP5008      sub-DP5012      sub-DP5017      sub-DP5021      sub-DP5025      sub-DP5029      sub-DP5033      sub-DP5037      sub-DP5044 sub-DP5005      sub-DP5009      sub-DP5013      sub-DP5018      sub-DP5022      sub-DP5026      sub-DP5030      sub-DP5034      sub-DP5038      sub-DP5047
sub-DP5002      sub-DP5006      sub-DP5010      sub-DP5014      sub-DP5019      sub-DP5023      sub-DP5027      sub-DP5031      sub-DP5035      sub-DP5040      sub-DP5050
sub-DP5003      sub-DP5007      sub-DP5011      sub-DP5016      sub-DP5020      sub-DP5024      sub-DP5028      sub-DP5032      sub-DP5036      sub-DP5041'



h_l='low'


task_high_low='StroopLow'

tasks='ses-StroopRealStim ses-StroopShamStim'

for sub in ${subjects};

do


for t in ${task_high_low};
do

for task in ${tasks};
do

bold_ref=/Volumes/823777_TNI_DP5_Healthy/DP5/xcp/fmriprep_out/fmriprep/${sub}/${task}/func/${sub}_${task}_task-${t}_run-1_space-T1w_boldref.nii.gz

if [[ -f "${bold_ref}" ]];

then
    mkdir -p /Volumes/823777_TNI_DP5_Healthy/DP5/Ben_SVR_fpcn_paper/mean_ts/${sub}/${task}/registrations/

    reg_dir=/Volumes/823777_TNI_DP5_Healthy/DP5/Ben_SVR_fpcn_paper/mean_ts/${sub}/${task}/registrations/

    ts_dir=/Volumes/823777_TNI_DP5_Healthy/DP5/Ben_SVR_fpcn_paper/mean_ts/${sub}/${task}

    seg=/Volumes/823777_TNI_DP5_Healthy/DP5/xcp/fmriprep_out/freesurfer/${sub}/mri/${sub}_yeo2011_seg.mgz




    ts=/Volumes/823777_TNI_DP5_Healthy/DP5/xcp/xcpOutput/task_data/pre/${h_l}/${sub}/${task}/task/stats/${sub}_${task}_res4d.nii.gz



    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    echo "Putting segmentation in proper orientation"
    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        mri_convert ${seg} \
        --in_orientation LIA \
        --out_orientation RAS \
        ${reg_dir}/${sub}_seg_oriented.nii.gz

    # Calculate affine and SyN warp
    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    echo "Procuring transformation matrix from segmentation to BOLD reference data"
    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        flirt -in ${reg_dir}/${sub}_seg_oriented.nii.gz \
        -ref ${bold_ref} -omat ${reg_dir}/${sub}_seg2func.mat -dof 12


    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    echo "Applying transformation matrix to segmentation image for ${sub}"
    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        flirt -in ${reg_dir}/${sub}_seg_oriented.nii.gz \
        -ref ${bold_ref} \
        -init ${reg_dir}/${sub}_seg2func.mat \
        -applyxfm \
        -out ${reg_dir}/${sub}_seg_in_func.nii.gz


    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    echo "Checking segmentation file and time series file dimensions"
    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    echo " Here is your segmentation file"

    mri_info ${reg_dir}/${sub}_seg_in_func.nii.gz


    echo "Here is the time-series file"

    mri_info ${ts}

    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    echo "Extracting time series"
    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # loop over level of task and create timeseries text files through the li 2019 parcellation
    for level in ${h_l};
    do



            echo "The timeseries file for ${sub} through the li 2019 parcellation does not exist."
            echo "Extracting timeseries through parcellation now"

            fslmeants -i ${ts} \
            -o ${ts_dir}/${sub}_${task}_${level}_li2019_ts.txt \
            --label=${reg_dir}/${sub}_seg_in_func.nii.gz

    done

fi

done
done
done