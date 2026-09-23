# Shared public-network dependency flags. macOS must use the OS library,
# never a Homebrew dylib that would be absent from a user's machine.
ifeq ($(OS),Windows_NT)
  PLATFORM_HTTP_CFLAGS := -DCURL_STATICLIB -Itools/curl-win/include
  PLATFORM_HTTP_LIBS := tools/curl-win/lib/libcurl.a -lwinhttp -lcrypt32 -lsecur32 -liphlpapi -lnormaliz -lws2_32 -lbcrypt
else ifeq ($(shell uname -s),Darwin)
  PLATFORM_HTTP_CFLAGS := -I$(shell xcrun --show-sdk-path)/usr/include
  PLATFORM_HTTP_LIBS := $(shell xcrun --show-sdk-path)/usr/lib/libcurl.tbd -framework CoreFoundation -framework CFNetwork
else
  PLATFORM_HTTP_CFLAGS := $(shell pkg-config --cflags libcurl)
  PLATFORM_HTTP_LIBS := $(shell pkg-config --libs libcurl) -ldl
endif
