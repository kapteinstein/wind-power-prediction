-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Jun 05, 2020 at 04:41 PM
-- Server version: 10.4.13-MariaDB-1:10.4.13+maria~bionic-log
-- PHP Version: 7.4.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `model_results`
--
CREATE DATABASE IF NOT EXISTS `model_results` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE `model_results`;

-- --------------------------------------------------------

--
-- Table structure for table `data`
--

DROP TABLE IF EXISTS `data`;
CREATE TABLE `data` (
  `data_id` int(11) NOT NULL,
  `data` blob DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `results`
--

DROP TABLE IF EXISTS `results`;
CREATE TABLE `results` (
  `id` int(11) NOT NULL,
  `region` varchar(64) DEFAULT NULL,
  `comment` varchar(255) DEFAULT NULL,
  `batch_size` int(11) DEFAULT NULL,
  `epochs` int(11) DEFAULT NULL,
  `window_size` int(11) DEFAULT NULL,
  `harmonics` tinyint(1) DEFAULT NULL,
  `hybrid` tinyint(1) DEFAULT NULL,
  `lightgbm` tinyint(1) DEFAULT NULL,
  `is_timestamps` tinyint(1) NOT NULL DEFAULT 0,
  `is_validation` tinyint(1) NOT NULL DEFAULT 0,
  `target` tinyint(1) DEFAULT NULL,
  `normalize` tinyint(1) DEFAULT NULL,
  `lr` float DEFAULT NULL,
  `lr_scheduler` tinyint(1) DEFAULT NULL,
  `optim` varchar(32) DEFAULT NULL,
  `ordinal` tinyint(1) DEFAULT NULL,
  `ordinal_resolution` int(11) DEFAULT NULL,
  `prosjektoppgave` tinyint(1) DEFAULT 0,
  `ratio_transform` int(11) DEFAULT NULL,
  `normalize_type` int(11) DEFAULT NULL,
  `data_length` int(11) NOT NULL,
  `dropout` float DEFAULT NULL,
  `master` tinyint(1) DEFAULT NULL,
  `seed` int(11) DEFAULT NULL,
  `cuda` tinyint(1) DEFAULT NULL,
  `shuffle` tinyint(1) DEFAULT NULL,
  `spp_resolution` int(11) DEFAULT NULL,
  `timestamp_end` timestamp NULL DEFAULT NULL,
  `timestamp_start` timestamp NULL DEFAULT NULL,
  `use_seed` tinyint(1) DEFAULT NULL,
  `verbose` int(11) DEFAULT 0,
  `git_commit` varchar(64) DEFAULT NULL,
  `created` timestamp NULL DEFAULT current_timestamp(),
  `updated` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `data_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `data`
--
ALTER TABLE `data`
  ADD PRIMARY KEY (`data_id`);

--
-- Indexes for table `results`
--
ALTER TABLE `results`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `data_id` (`data_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `data`
--
ALTER TABLE `data`
  MODIFY `data_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `results`
--
ALTER TABLE `results`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `results`
--
ALTER TABLE `results`
  ADD CONSTRAINT `results_data_fk` FOREIGN KEY (`data_id`) REFERENCES `data` (`data_id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
