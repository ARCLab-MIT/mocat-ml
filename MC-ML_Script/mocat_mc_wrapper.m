function mocat_mc_wrapper(ICfile, seed)
    % Add necessary paths
    %addpath(genpath('C:/Users/Nathan/Desktop/mocat-ml-main/MC-ML_Script/'));
    %addpath(genpath('C:/Users/Nathan/Desktop/mocat-ml-main/MC-ML_Script/supporting_data/'));
    %addpath(genpath('C:/Users/Nathan/Desktop/mocat-ml-main/MC-ML_Script/supporting_functions/'));
    addpath(genpath('/Users/woodywu/Desktop/Research/Project_orbitalrisk/MOCAT_ML/mocat-ml-nathan_lev/MC-ML_Script/'));
    addpath(genpath('/Users/woodywu/Desktop/Research/Project_orbitalrisk/MOCAT_ML/mocat-ml-nathan_lev/MC-ML_Script/supporting_data/'));
    addpath(genpath('/Users/woodywu/Desktop/Research/Project_orbitalrisk/MOCAT_ML/mocat-ml-nathan_lev/MC-ML_Script/supporting_functions/'));

    % Set initial conditions and seed
    params.ICfile = ICfile;
    params.seed = seed;

    % Run the Quick_Start script
    Quick_Start;

    % Print or save the output file path
    output_file_path = cfgMC.filename_save;
    fprintf('MOCAT-MC simulation complete. Data saved to %s\n', output_file_path);
    
    % Save the output file path to a temporary text file
    fileID = fopen('mocat_output_path.txt', 'w');
    fprintf(fileID, '%s\n', output_file_path);
    fclose(fileID);
end

