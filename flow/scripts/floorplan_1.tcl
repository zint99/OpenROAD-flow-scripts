utl::set_metrics_stage "floorplan__{}"
source $::env(SCRIPTS_DIR)/load.tcl
erase_non_stage_variables floorplan
load_design 1_synth.v 1_synth.sdc
