#!/bin/bash
export PATH=$PATH:/usr/local/go/bin
export PATH=$PATH:$(go env GOPATH)/bin
COMCAST="/root/go/bin/comcast" # Comcast 的路径
INTERFACE=lo                  # 第一个网络接口
PORT_1=8000                     # 第一个端口

FILE=$1
if [ -z "$FILE" ]; then
  echo "Policy file name has to be specified"
  exit 1
fi

parsePolicyFile() {
  filename=$1
  if [ -z "$filename" ]; then
    echo "Filename parameter is required"
    exit 1
  fi

  latestLoss="0"
  latestDelay="0"
  latestRate="0"

  while read -r line; do
    if [[ $line == \#* ]]; then
      continue;
    fi

    keys=($line)
    comm=${keys[0]}
    value=${keys[1]}

    case $comm in
      rate)
        latestRate=$value
        echo "Setting rate to $latestRate"
        ;;
      loss)
        latestLoss=$value
        echo "Setting packet loss to $latestLoss"
        ;;
      delay)
        latestDelay=$value
        echo "Setting delay to $latestDelay"
        ;;
      wait)
        echo "Waiting for $value seconds"
        sleep "$value"
        ;;
    esac
    
    $COMCAST --device "$INTERFACE" --stop
    # 应用当前的网络限制
    $COMCAST --device "$INTERFACE" \
             --target-port "$PORT_1"\
             --latency "$latestDelay" \
             --packet-loss "$latestLoss" \
             --target-bw "$latestRate"
  done < "$filename"
}

policyLoop() {
  filename=$1
  while true; do
    parsePolicyFile "$filename"
  done
}

policyLoop "$FILE" 
