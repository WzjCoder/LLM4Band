#!/bin/bash

runTestsOnModel() {
  modelName=$1
  for PROFILE in cascade_profile low_bw high_bw
  do
    for ((runIndex=1;runIndex<=1;runIndex++)); 
    do
      # clear the logs
      # rm -rf webrtc.log
      # rm -rf bandwidth_estimator.log
      # rm -rf plot.png

      # set active model
      # MODEL="${MODELDIR}/${modelName}"
      # echo $MODEL >active_model
      sleep 1
      # docker run -d --rm -v `pwd`:/app -w /app --name alphartc_pyinfer --cap-add=NET_ADMIN challenge-env peerconnection_serverless receiver_pyinfer.json
      # sleep 1
      
      rm "receiver_log/${modelName}/receiver_${PROFILE}_${runIndex}.log"
      rm "logging/webrtc_comcast.log"
      peerconnection_serverless receiver_pyinfer_offline.json > receiver_log/${modelName}/receiver_${PROFILE}_${runIndex}.log 2>&1 &
      sleep 3

      rm "comcast_log/${modelName}/output_${PROFILE}_${runIndex}.log"
      bash ./comcast_profiles/comcast_policy.sh comcast_profiles/$PROFILE  > comcast_log/${modelName}/output_${PROFILE}_${runIndex}.log 2>&1 &
      peerconnection_serverless sender_pyinfer.json 
      sleep 1

      averageBW=`python3 calculateAverageBandwidth.py --network_profile comcast_profiles/${PROFILE} | sed 's/averageBandwidth: \(.*\)/\1/'`
      maxdelay=500
      cd metrics/
      python3 eval_network.py --dst_network_log /app/logging/webrtc_comcast.log --output /app/result/${modelName}/result_comcast_${PROFILE}_${runIndex}.json --ground_recv_rate $averageBW --max_delay $maxdelay
      cd ..
      # bash ./eval.sh $averageBW maxdelay
      # python3 plot.py --title $PROFILE
      # mv plot.png $resultsDir/plot_${modelName}_${PROFILE}_${runIndex}.png
      # python3 log_eval.py --model ${modelName} --network_profile ${PROFILE} --eval_log_file ${EVAL_LOGFILE}
      
      # move evaluation results to resultdir
      # mv out_eval_video.json ${resultsDir}/out_eval_video_${modelName}_${PROFILE}_${runIndex}.json
      # mv out_eval_network.json ${resultsDir}/out_eval_network_${modelName}_${PROFILE}_${runIndex}.json
      # mv bandwidth_estimator.log ${resultsDir}/bandwidth_estimator_${modelName}_${PROFILE}_${runIndex}.log
      # mv outvideo.yuv ${resultsDir}/outputvideo_${modelName}_${PROFILE}_${runIndex}.yuv
      # mv outaudio.wav ${resultsDir}/outaudio_${modelName}_${PROFILE}_${runIndex}.wav
    done
  done
}

# 创建字符串列表（以空格分隔）
models="mlp encoder_bwhead gpt2_only gpt2 gpt2_lora_only gpt2_lora"

# 遍历字符串列表
for model in $models; do
    echo "Currently processing model: $model"
    mkdir -p "comcast_log/${model}"
    mkdir -p "receiver_log/${model}"
    cp BandwidthEstimator_${model}.py BandwidthEstimator.py
    runTestsOnModel "$model"
done