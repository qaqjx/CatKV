#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>


namespace disk_manager {

class ThreadPool {
public:
    explicit ThreadPool(size_t max_workers)
        : stop_(false) {
        if (max_workers == 0) {
            max_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
            if (max_workers == 0) {
                max_workers = 4;
            }
        }
        for (size_t i = 0; i < max_workers; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    template <class F>
    auto submit(F&& task)
        -> std::future<typename std::invoke_result_t<F>> {
        using ResultT = typename std::invoke_result_t<F>;
        auto packaged = std::make_shared<std::packaged_task<ResultT()>>(std::forward<F>(task));
        std::future<ResultT> fut = packaged->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks_.emplace([packaged]() { (*packaged)(); });
        }
        cv_.notify_one();
        return fut;
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

}  // namespace disk_manager