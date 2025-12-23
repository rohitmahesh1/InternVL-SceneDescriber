Shift-Command-3 for the whole screen, Shift-Command-4 to drag a selection, or Shift-Command-5 for a toolbar with more options (like window or screen recording)

IMG_B64=$(base64 < "$HOME/Downloads/tiger.jpeg" | tr -d '\n')
grpcurl -plaintext -d '{ "image": "'"$IMG_B64"'", "filename": "tiger.jpeg", "question":"what is this animal doing and where might it be?" }' localhost:50051 scenedescriber.SceneService/AskAboutScene

IMG_DISH=$(base64 < "$HOME/Downloads/dish.png" | tr -d '\n')
grpcurl -plaintext -d '{ "image": "'"$IMG_DISH"'", "filename": "dish.png", "question":"what is the white thing in the dish?" }' \
localhost:50051 scenedescriber.SceneService/AskAboutScene

setting up protobuf tools for swift
brew install protobuf
brew install swift-protobuf grpc-swift

Bulk comment
select all the text
Press ⌘/
That toggles line comments (//) on every selected line.
To undo it, select all again and press ⌘/ once more.

Screenshot
shift-command-4