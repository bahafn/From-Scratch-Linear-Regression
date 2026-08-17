CC := gcc

CFLAGS := -Wall -Wextra -Wpedantic
LDLIBS := -lm

SRC     := src/matrix.c src/linear_regression.c
APP_SRC := src/main.c $(SRC)

BUILD_DIR := build

RELEASE_DIR := $(BUILD_DIR)/release
DEBUG_DIR   := $(BUILD_DIR)/debug
TEST_DIR    := $(BUILD_DIR)/test

RELEASE_BIN := $(RELEASE_DIR)/regression
DEBUG_BIN   := $(DEBUG_DIR)/regression
TEST_BIN    := $(TEST_DIR)/tests

.PHONY: all release debug test clean

all: release

release: $(RELEASE_BIN)

debug: $(DEBUG_BIN)

test: $(TEST_BIN)
	$(TEST_BIN)

$(RELEASE_BIN): $(APP_SRC) | $(RELEASE_DIR)
	$(CC) $(CFLAGS) -O3 -flto $(APP_SRC) $(LDLIBS) -o $@

$(DEBUG_BIN): $(APP_SRC) | $(DEBUG_DIR)
	$(CC) $(CFLAGS) -O0 -g $(APP_SRC) $(LDLIBS) -o $@

$(TEST_BIN): $(SRC) tests/*.c | $(TEST_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) \
		$(SRC) \
		tests/*.c \
		-lcunit \
		$(LDLIBS) \
		-o $@

$(RELEASE_DIR) $(DEBUG_DIR) $(TEST_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
