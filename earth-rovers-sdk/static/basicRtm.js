$(document).ready(function () {
  window.rtm_connection_state = "INITIALIZING";
  window.rtm_channel_state = "NOT_JOINED";
  window.rtm_last_error = null;
  window.rtm_last_send_state = "NOT_SENT";
  window.rtm_ready = false;
  const APP_ID = $("#appid").val();
  const USER_ID = $("#uid").val();
  const TOKEN = $("#rtm_token").val();
  const channelName = $("#channel").val(); // Replace with your desired channel name
  const botUid = $("#bot_uid").val();

  // Create an instance of the Agora RTM client
  const rtmClient = AgoraRTM.createInstance(APP_ID);

  // Create an instance of an RTM channel
  const rtmChannel = rtmClient.createChannel(channelName);

  // Event listener for connection state changes
  rtmClient.on("ConnectionStateChange", (newState, reason) => {
    window.rtm_connection_state = `${newState}:${reason}`;
    if (newState === "CONNECTED") {
      window.rtm_ready = true;
    } else if (newState === "DISCONNECTED" || newState === "ABORTED") {
      window.rtm_ready = false;
    }
    console.log(
      "on connection state changed to " + newState + " reason: " + reason
    );
  });

  // Event listener for receiving a channel message
  rtmChannel.on("ChannelMessage", ({ text }, senderId) => {
    console.log("AgoraRTM msg from user " + senderId + " received: \n" + text);
  });

  rtmClient.on("MessageFromPeer", function (message, peerId) {
    const controls = JSON.parse(new TextDecoder().decode(message.rawMessage));
    if (controls.timestamp !== undefined && controls.source_timestamp === undefined) {
      controls.source_timestamp = controls.timestamp;
    }
    const event = new CustomEvent("message-from-peer", { detail: controls });
    window.rtm_data = controls;
    window.rtm_data.timestamp = Number(Date.now() / 1000);
    window.rtm_data.browser_received_timestamp = window.rtm_data.timestamp;
    const formattedMessage = formatMessage(controls);
    $("#messages").html(formattedMessage);
    document.dispatchEvent(event);

    if (controls.latitude && controls.longitude) {
      const latitude = parseFloat(controls.latitude);
      const longitude = parseFloat(controls.longitude);
      console.log("updating marker");
      window.updateMarker(latitude, longitude);
    }

    console.log(
      "AgoraRTM peer msg from user " + peerId + " received: \n",
      controls
    );
  });

  function formatMessage(jsonData) {
    let formattedMessage =
      '<div class="card"><div class="card-body"><h5 class="card-title">Message from Peer</h5><ul class="list-group">';
    for (const [key, value] of Object.entries(jsonData)) {
      formattedMessage += `<li class="list-group-item"><strong>${key}:</strong> ${value}</li>`;
    }
    formattedMessage += "</ul></div></div>";
    return formattedMessage;
  }

  // Function to join the RTM channel
  function joinRTMChannel(uid) {
    rtmClient
      .login({ token: TOKEN, uid: String(uid) })
      .then(() => {
        window.rtm_connection_state = "LOGGED_IN";
        window.rtm_ready = true;
        window.rtm_last_error = null;
        console.log("AgoraRTM client login success");
        // Join a channel
        rtmChannel
          .join()
          .then(() => {
            window.rtm_channel_state = "JOINED";
            console.log("RTM Channel join success");
            // You can now send messages or set up more event listeners
          })
          .catch((error) => {
            window.rtm_channel_state = "JOIN_FAILED";
            window.rtm_last_error = String(error).slice(0, 240);
            console.log("Failed to join channel for error: " + error);
          });
      })
      .catch((err) => {
        window.rtm_connection_state = "LOGIN_FAILED";
        window.rtm_ready = false;
        window.rtm_last_error = String(err).slice(0, 240);
        console.log("AgoraRTM client login failure", err);
      });
  }

  // Function to send message
  async function sendMessage(json) {
    if (!window.rtm_ready) {
      const error = new Error(
        `RTM control bridge is not logged in (${window.rtm_connection_state})`
      );
      window.rtm_last_send_state = "NOT_READY";
      window.rtm_last_error = error.message;
      throw error;
    }
    const message = JSON.stringify(json);
    console.warn("sending message to bot", botUid);
    console.warn("message", message);
    window.rtm_last_send_state = "SENDING";
    return rtmClient
      .sendMessageToPeer(
        {
          text: message,
        },
        botUid
      )
      .then(() => {
        window.rtm_last_send_state = "SENT";
        window.rtm_last_error = null;
        console.warn("Message sent successfully:", message);
      })
      .catch((err) => {
        window.rtm_last_send_state = "FAILED";
        window.rtm_last_error = String(err).slice(0, 240);
        console.warn("Error sending message:", err);
        throw err;
      });
  }

  // Make the function globally accessible
  window.sendMessage = sendMessage;

  // Call joinRTMChannel with the user ID to start the process
  joinRTMChannel(USER_ID);
});
